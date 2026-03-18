import inspect
import logging
import os
import queue
import threading
import warnings
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    DynamicLayer,
    EncoderDecoderCache,
    QuantizedCache,
    StaticCache,
)
from transformers.generation.configuration_utils import (
    ALL_STATIC_CACHE_IMPLEMENTATIONS,
    DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS,
    STATIC_CACHE_IMPLEMENTATIONS,
    GenerationConfig,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GENERATION_MODES_MAPPING,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    GenerateOutput,
    GenerationMixin,
    GenerationMode,
    TransformersKwargs,
)
from transformers.utils import is_hqq_available, is_optimum_quanto_available

logger = logging.getLogger(__name__)

class DynamicCache(DynamicCache):

    def __init__(self) -> None:
        super().__init__()

    @property
    def key_cache(self):
        return [layer.keys for layer in self.layers]

    @key_cache.setter
    def key_cache(self, value):
        if len(value) != len(self.layers):
            self.layers = [DynamicLayer() for _ in value]
        for layer, keys in zip(self.layers, value):
            layer.keys = keys
            if not layer.is_initialized and keys.numel() > 0:
                layer.dtype = keys.dtype
                layer.device = keys.device
                layer.is_initialized = True

    @property
    def value_cache(self):
        return [layer.values for layer in self.layers]

    @value_cache.setter
    def value_cache(self, value):
        if len(value) != len(self.layers):
            self.layers = [DynamicLayer() for _ in value]
        for layer, values in zip(self.layers, value):
            layer.values = values
            if not layer.is_initialized and values.numel() > 0:
                layer.dtype = values.dtype
                layer.device = values.device
                layer.is_initialized = True

    def pop(self):
        for layer in self.layers:
            if layer.is_initialized and layer.get_seq_length() > 0:
                layer.keys = layer.keys[..., :-1, :]
                layer.values = layer.values[..., :-1, :]

class unified_PreTrainedModel(PreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return self._generate_stream(
            inputs, generation_config, logits_processor, stopping_criteria,
            prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer,
            negative_prompt_ids, negative_prompt_attention_mask, **kwargs,
        )

    def rebuild_past_from_source_target(self):
        self.past_key_values.layers = []
        source_len = self.source_key_values.get_seq_length() if self.source_key_values else 0
        target_len = self.target_key_values.get_seq_length() if self.target_key_values else 0

        if source_len > 0:
            for i, source_layer in enumerate(self.source_key_values.layers):
                new_layer = DynamicLayer()
                if source_layer.is_initialized:
                    if target_len > 0 and i < len(self.target_key_values.layers):
                        target_layer = self.target_key_values.layers[i]
                        if target_layer.is_initialized:
                            new_layer.keys = torch.cat([source_layer.keys, target_layer.keys], dim=2)
                            new_layer.values = torch.cat([source_layer.values, target_layer.values], dim=2)
                        else:
                            new_layer.keys = source_layer.keys.clone()
                            new_layer.values = source_layer.values.clone()
                    else:
                        new_layer.keys = source_layer.keys.clone()
                        new_layer.values = source_layer.values.clone()
                    new_layer.dtype = source_layer.dtype
                    new_layer.device = source_layer.device
                    new_layer.is_initialized = True
                self.past_key_values.layers.append(new_layer)

    def merge_source_target(self):
        self.rebuild_past_from_source_target()

    def separate_source_target(self):
        self.rebuild_past_from_source_target()

    def _generate_stream(
        self,
        inputs, generation_config, logits_processor, stopping_criteria,
        prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer,
        negative_prompt_ids, negative_prompt_attention_mask, **kwargs,
    ):
        self.model.reset_video_cache()

        trust_remote_code = kwargs.pop("trust_remote_code", None)
        if isinstance(kwargs.get("custom_generate"), str):
            raise NotImplementedError("custom_generate from Hub is not supported in streaming mode.")

        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            None, kwargs, synced_gpus, assistant_model, streamer,
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, None, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)

        use_parallel = os.environ.get("STREAM_PARALLEL", "0") == "1" or model_kwargs.pop("use_parallel", False)
        if use_parallel and hasattr(self, "_sample_stream_parallel"):
            decoding_method = getattr(type(self), "_sample_stream_parallel")
        else:
            decoding_method = getattr(type(self), "_sample_stream")

        split_k = model_kwargs.pop("split_k", None)
        end_Instruct = model_kwargs.pop("end_Instruct", None)
        assistant_token = model_kwargs.pop("assistant_token", None)
        _lengths = model_kwargs.pop("_lengths", None)
        _lengths_index = model_kwargs.pop("_lengths_index", None)
        target_start_pos = model_kwargs.pop("target_start_pos", 0)
        seg_think_Instruct_token = model_kwargs.pop("seg_think_Instruct_token", None)
        q_think_Instruct_token = model_kwargs.pop("q_think_Instruct_token", None)
        _ = model_kwargs.pop("use_parallel", None)

        self._validate_model_kwargs_stream(model_kwargs)
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)

        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        max_cache_length = generation_config.max_length - 1
        if inputs_tensor.shape[1] != input_ids_length and model_input_name == "inputs_embeds" and not self.config.is_encoder_decoder:
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["tokenizer"] = generation_mode_kwargs.pop("tokenizer", None)

        result = decoding_method(
            self, input_ids,
            _lengths=_lengths,
            _lengths_index=_lengths_index,
            assistant_token=assistant_token,
            seg_think_Instruct_token=seg_think_Instruct_token,
            q_think_Instruct_token=q_think_Instruct_token,
            end_Instruct=end_Instruct,
            target_start_pos=target_start_pos,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        )

        return result

    def _sample_stream(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        split_k: Optional[int] = None,
        _lengths: Optional[dict] = None,
        _lengths_index: Optional[torch.Tensor] = None,
        end_Instruct: Optional[str] = None,
        assistant_token: Optional[torch.Tensor] = None,
        seg_think_Instruct_token: Optional[torch.Tensor] = None,
        q_think_Instruct_token: Optional[torch.Tensor] = None,
        target_start_pos: int = 0,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        self.source_key_values = DynamicCache()
        self.target_key_values = DynamicCache()
        self.past_key_values = DynamicCache()
        last_boundary_token = None
        tokenizer = model_kwargs.pop("tokenizer", None)

        model_kwargs["assistant_token"] = assistant_token
        model_kwargs["target_start_pos"] = target_start_pos
        model_kwargs["_lengths"] = _lengths
        model_kwargs["_lengths_index"] = _lengths_index
        model_kwargs["seg_think_Instruct_token"] = seg_think_Instruct_token
        model_kwargs["q_think_Instruct_token"] = q_think_Instruct_token

        source_seg_len = _lengths[0]["source_seg_len"]
        segment_types = _lengths[0].get("segment_types", [])
        total_segments = len(source_seg_len)

        EOT_id = tokenizer.convert_tokens_to_ids("<EOT>") if tokenizer else 151670
        EOQ_id = tokenizer.convert_tokens_to_ids("<EOQ>") if tokenizer else 151669
        eos_token_id = tokenizer.eos_token_id if tokenizer else None
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") if tokenizer else None

        current_seg_idx = 0
        ReadAction = True
        need_think_prefix = True
        need_assistant_prefix = True
        current_seg_type = None

        next_tokens = assistant_token.unsqueeze(0)
        target_tokens = [next_tokens]

        source_input_length = sum(source_seg_len[: current_seg_idx + 1])
        target_input_length = 1
        input_length = (source_input_length, target_input_length)

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):

            if ReadAction:
                model_inputs = self.prepare_inputs_for_generation_stream(
                    input_ids, input_length=input_length, ReadAction=ReadAction,
                    is_streaming=True, **model_kwargs,
                )
                _outputs = self(**model_inputs, return_dict=True)

                if hasattr(_outputs, "source_key_values") and _outputs.source_key_values is not None:
                    self.source_key_values = _outputs.source_key_values

                ReadAction = False
                need_think_prefix = True
                self.merge_source_target()

            else:
                if need_think_prefix:
                    self.separate_source_target()
                    current_type = segment_types[current_seg_idx] if current_seg_idx < len(segment_types) else "video"
                    current_seg_type = current_type
                    instruct_token = q_think_Instruct_token if current_type == "text" else seg_think_Instruct_token

                    if need_assistant_prefix:
                        if instruct_token is not None:
                            instruct_token = instruct_token.to(input_ids.device)
                            if instruct_token.dim() == 1:
                                instruct_token = instruct_token.unsqueeze(0)
                            write_tokens = torch.cat([next_tokens, instruct_token], dim=-1)
                            target_tokens.append(instruct_token)
                            input_ids = torch.cat([input_ids, next_tokens, instruct_token], dim=-1)
                        else:
                            write_tokens = next_tokens
                            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                        need_assistant_prefix = False
                    else:
                        if instruct_token is not None:
                            instruct_token = instruct_token.to(input_ids.device)
                            if instruct_token.dim() == 1:
                                instruct_token = instruct_token.unsqueeze(0)
                            if last_boundary_token is not None:
                                boundary_tensor = torch.tensor(
                                    [[last_boundary_token]], dtype=torch.long, device=input_ids.device
                                )
                                write_tokens = torch.cat([boundary_tensor, instruct_token], dim=-1)
                            else:
                                write_tokens = instruct_token
                            target_tokens.append(write_tokens)
                            input_ids = torch.cat([input_ids, write_tokens], dim=-1)
                        else:
                            write_tokens = next_tokens

                    need_think_prefix = False
                else:
                    write_tokens = next_tokens

                if isinstance(write_tokens, torch.Tensor) and write_tokens.dim() == 3 and write_tokens.size(0) == 1:
                    write_tokens = write_tokens.squeeze(0)

                model_inputs = self.prepare_inputs_for_generation_stream(
                    write_tokens, input_length=input_length, ReadAction=ReadAction,
                    is_streaming=True, **model_kwargs,
                )

                outputs = self(**model_inputs, return_dict=True)

                if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    output_cache = outputs.past_key_values
                    if isinstance(output_cache, tuple):
                        output_cache = DynamicCache.from_legacy_cache(output_cache)
                    source_len = self.source_key_values.get_seq_length() if self.source_key_values else 0
                    total_len = output_cache.get_seq_length() if output_cache else 0
                    if total_len > source_len:
                        self.target_key_values.layers = []
                        for layer in output_cache.layers:
                            if layer.is_initialized:
                                target_layer = DynamicLayer()
                                target_layer.keys = layer.keys[..., source_len:, :]
                                target_layer.values = layer.values[..., source_len:, :]
                                target_layer.dtype = layer.dtype
                                target_layer.device = layer.device
                                target_layer.is_initialized = True
                                self.target_key_values.layers.append(target_layer)
                    self.past_key_values = output_cache

                if synced_gpus and this_peer_finished:
                    continue

                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                del outputs

                generated_token_id = next_tokens.item()
                is_segment_boundary = (generated_token_id == EOT_id) or (generated_token_id == EOQ_id)
                is_final_end = (generated_token_id == im_end_id) or (generated_token_id == eos_token_id)

                if is_segment_boundary:
                    expected = EOQ_id if current_seg_type == "text" else EOT_id
                    if expected is not None and generated_token_id != expected:
                        generated_token_id = expected
                        next_tokens = torch.tensor(
                            [generated_token_id], device=input_ids.device, dtype=torch.long
                        ).unsqueeze(0)

                if not is_segment_boundary:
                    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                    if streamer is not None:
                        streamer.put(next_tokens.cpu())
                    if next_tokens.dim() == 1:
                        next_tokens = next_tokens.unsqueeze(0)
                    target_tokens.append(next_tokens)
                else:
                    if next_tokens.dim() == 1:
                        next_tokens = next_tokens.unsqueeze(0)
                    if streamer is not None:
                        streamer.put(next_tokens[0].cpu())

                target_ids = torch.cat(target_tokens, dim=-1)

                if is_segment_boundary:
                    current_seg_idx += 1
                    if current_seg_idx < total_segments:
                        last_boundary_token = generated_token_id
                        ReadAction = True
                        source_input_length = sum(source_seg_len[: current_seg_idx + 1])
                        input_length = (source_input_length, target_input_length)
                    else:
                        boundary_tensor = torch.tensor(
                            [[generated_token_id]], dtype=torch.long, device=input_ids.device
                        )
                        target_tokens.append(boundary_tensor)
                        input_ids = torch.cat([input_ids, boundary_tensor], dim=-1)
                        target_ids = torch.cat(target_tokens, dim=-1)
                        ReadAction = False

                elif is_final_end:
                    unfinished_sequences = unfinished_sequences * 0
                    this_peer_finished = True

                target_generated_len = target_ids.shape[-1] - 2
                max_new = generation_config.max_new_tokens if generation_config.max_new_tokens is not None else float("inf")
                if target_generated_len >= max_new and not is_segment_boundary:
                    unfinished_sequences = unfinished_sequences * 0
                    this_peer_finished = True

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=target_ids, scores=scores, logits=raw_logits,
                attentions=decoder_attentions, hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return target_ids

    def _sample_stream_parallel(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        split_k: Optional[int] = None,
        _lengths: Optional[dict] = None,
        _lengths_index: Optional[torch.Tensor] = None,
        end_Instruct: Optional[str] = None,
        assistant_token: Optional[torch.Tensor] = None,
        seg_think_Instruct_token: Optional[torch.Tensor] = None,
        q_think_Instruct_token: Optional[torch.Tensor] = None,
        target_start_pos: int = 0,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        stream_debug = os.environ.get("STREAM_DEBUG", "0") == "1"

        self.segment_source_caches = {}
        self.segment_read_complete = {}
        self.read_thread_exception = None
        self.read_stop_flag = threading.Event()

        self.source_key_values = DynamicCache()
        self.target_key_values = DynamicCache()
        self.past_key_values = DynamicCache()

        tokenizer = model_kwargs.pop("tokenizer", None)

        model_kwargs["assistant_token"] = assistant_token
        model_kwargs["target_start_pos"] = target_start_pos
        model_kwargs["_lengths"] = _lengths
        model_kwargs["_lengths_index"] = _lengths_index
        model_kwargs["seg_think_Instruct_token"] = seg_think_Instruct_token
        model_kwargs["q_think_Instruct_token"] = q_think_Instruct_token

        source_seg_len = _lengths[0]["source_seg_len"]
        segment_types = _lengths[0].get("segment_types", [])
        total_segments = len(source_seg_len)

        EOT_id = tokenizer.convert_tokens_to_ids("<EOT>") if tokenizer else 151670
        EOQ_id = tokenizer.convert_tokens_to_ids("<EOQ>") if tokenizer else 151669
        eos_token_id = tokenizer.eos_token_id if tokenizer else None
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>") if tokenizer else None

        for seg_idx in range(total_segments):
            self.segment_read_complete[seg_idx] = threading.Event()

        source_cache_lock = threading.Lock()

        device = input_ids.device
        read_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

        def prefetch_segment_async(seg_idx: int, local_source_cache: DynamicCache):
            if seg_idx >= total_segments:
                return local_source_cache

            cumulative_len = sum(source_seg_len[: seg_idx + 1])

            if stream_debug:
                try:
                    past_len = local_source_cache.get_seq_length() if local_source_cache else 0
                except Exception:
                    past_len = 0
                cur_type = segment_types[seg_idx] if seg_idx < len(segment_types) else "na"
                to_read = max(0, cumulative_len - past_len)
                print(
                    f"[PARALLEL_READ] seg={seg_idx}/{total_segments-1} type={cur_type} "
                    f"past_len={past_len} cumulative_len={cumulative_len} to_read={to_read}",
                    flush=True,
                )

            read_input_length = (cumulative_len, 0)

            original_source = self.source_key_values
            self.source_key_values = local_source_cache

            try:
                read_model_kwargs = model_kwargs.copy()
                past_source_len = local_source_cache.get_seq_length() if local_source_cache else 0
                read_model_kwargs["cache_position"] = torch.tensor([past_source_len], device=input_ids.device)

                model_inputs = self.prepare_inputs_for_generation_stream(
                    input_ids,
                    input_length=read_input_length,
                    ReadAction=True,
                    is_streaming=True,
                    **read_model_kwargs,
                )

                if read_stream is not None:
                    with torch.cuda.stream(read_stream):
                        with torch.no_grad():
                            _outputs = self(**model_inputs, return_dict=True)
                    read_stream.synchronize()
                else:
                    with torch.no_grad():
                        _outputs = self(**model_inputs, return_dict=True)

                if hasattr(_outputs, "source_key_values") and _outputs.source_key_values is not None:
                    local_source_cache = _outputs.source_key_values

            finally:
                self.source_key_values = original_source

            return local_source_cache

        def clone_cache(cache: DynamicCache) -> DynamicCache:
            new_cache = DynamicCache()
            if cache and cache.get_seq_length() > 0:
                new_cache.layers = []
                for layer in cache.layers:
                    if layer.is_initialized:
                        new_layer = DynamicLayer()
                        new_layer.keys = layer.keys.clone()
                        new_layer.values = layer.values.clone()
                        new_layer.dtype = layer.dtype
                        new_layer.device = layer.device
                        new_layer.is_initialized = True
                        new_cache.layers.append(new_layer)
            return new_cache

        def read_worker():
            try:
                local_source_cache = DynamicCache()

                for seg_idx in range(total_segments):
                    if self.read_stop_flag.is_set():
                        break

                    local_source_cache = prefetch_segment_async(seg_idx, local_source_cache)

                    seg_cache = clone_cache(local_source_cache)

                    with source_cache_lock:
                        self.segment_source_caches[seg_idx] = seg_cache

                    self.segment_read_complete[seg_idx].set()

                    if stream_debug:
                        print(
                            f"[PARALLEL_READ] seg={seg_idx} COMPLETE, "
                            f"cache_len={seg_cache.get_seq_length() if seg_cache else 0}",
                            flush=True,
                        )

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.read_thread_exception = e
                for idx in range(total_segments):
                    if not self.segment_read_complete[idx].is_set():
                        self.segment_read_complete[idx].set()

        read_thread = threading.Thread(target=read_worker, daemon=True, name="ParallelReadWorker")
        read_thread.start()

        current_seg_idx = 0
        last_boundary_token = None
        need_think_prefix = True
        need_assistant_prefix = True
        current_seg_type = None

        next_tokens = assistant_token.unsqueeze(0)
        target_tokens = [next_tokens]

        source_input_length = sum(source_seg_len[: current_seg_idx + 1])
        target_input_length = 1
        input_length = (source_input_length, target_input_length)

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)

        need_wait_for_segment = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):

            if self.read_thread_exception is not None:
                raise self.read_thread_exception

            if need_wait_for_segment:
                if stream_debug:
                    print(f"[PARALLEL_WRITE] Waiting for seg={current_seg_idx}...", flush=True)

                self.segment_read_complete[current_seg_idx].wait()

                if self.read_thread_exception is not None:
                    raise self.read_thread_exception

                with source_cache_lock:
                    if current_seg_idx in self.segment_source_caches:
                        self.source_key_values = self.segment_source_caches[current_seg_idx]

                if stream_debug:
                    src_len = self.source_key_values.get_seq_length() if self.source_key_values else 0
                    print(f"[PARALLEL_WRITE] seg={current_seg_idx} ready, source_len={src_len}", flush=True)

                self.merge_source_target()

                need_wait_for_segment = False
                need_think_prefix = True

            if need_think_prefix:
                self.separate_source_target()

                current_type = segment_types[current_seg_idx] if current_seg_idx < len(segment_types) else "video"
                current_seg_type = current_type
                instruct_token = q_think_Instruct_token if current_type == "text" else seg_think_Instruct_token

                if need_assistant_prefix:
                    if instruct_token is not None:
                        instruct_token = instruct_token.to(input_ids.device)
                        if instruct_token.dim() == 1:
                            instruct_token = instruct_token.unsqueeze(0)
                        write_tokens = torch.cat([next_tokens, instruct_token], dim=-1)
                        target_tokens.append(instruct_token)
                        input_ids = torch.cat([input_ids, next_tokens, instruct_token], dim=-1)
                    else:
                        write_tokens = next_tokens
                        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    need_assistant_prefix = False
                else:
                    if instruct_token is not None:
                        instruct_token = instruct_token.to(input_ids.device)
                        if instruct_token.dim() == 1:
                            instruct_token = instruct_token.unsqueeze(0)
                        if last_boundary_token is not None:
                            boundary_tensor = torch.tensor(
                                [[last_boundary_token]], dtype=torch.long, device=input_ids.device
                            )
                            write_tokens = torch.cat([boundary_tensor, instruct_token], dim=-1)
                        else:
                            write_tokens = instruct_token
                        target_tokens.append(write_tokens)
                        input_ids = torch.cat([input_ids, write_tokens], dim=-1)
                    else:
                        write_tokens = next_tokens

                need_think_prefix = False
            else:
                write_tokens = next_tokens

            if isinstance(write_tokens, torch.Tensor) and write_tokens.dim() == 3 and write_tokens.size(0) == 1:
                write_tokens = write_tokens.squeeze(0)

            model_inputs = self.prepare_inputs_for_generation_stream(
                write_tokens, input_length=input_length, ReadAction=False,
                is_streaming=True, **model_kwargs,
            )

            outputs = self(**model_inputs, return_dict=True)

            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                output_cache = outputs.past_key_values
                if isinstance(output_cache, tuple):
                    output_cache = DynamicCache.from_legacy_cache(output_cache)
                source_len = self.source_key_values.get_seq_length() if self.source_key_values else 0
                total_len = output_cache.get_seq_length() if output_cache else 0
                if total_len > source_len:
                    self.target_key_values.layers = []
                    for layer in output_cache.layers:
                        if layer.is_initialized:
                            target_layer = DynamicLayer()
                            target_layer.keys = layer.keys[..., source_len:, :]
                            target_layer.values = layer.values[..., source_len:, :]
                            target_layer.dtype = layer.dtype
                            target_layer.device = layer.device
                            target_layer.is_initialized = True
                            self.target_key_values.layers.append(target_layer)
                self.past_key_values = output_cache

            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            del outputs

            generated_token_id = next_tokens.item()
            is_segment_boundary = (generated_token_id == EOT_id) or (generated_token_id == EOQ_id)
            is_final_end = (generated_token_id == im_end_id) or (generated_token_id == eos_token_id)

            if is_segment_boundary:
                expected = EOQ_id if current_seg_type == "text" else EOT_id
                if expected is not None and generated_token_id != expected:
                    generated_token_id = expected
                    next_tokens = torch.tensor(
                        [generated_token_id], device=input_ids.device, dtype=torch.long
                    ).unsqueeze(0)

            if not is_segment_boundary:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                if next_tokens.dim() == 1:
                    next_tokens = next_tokens.unsqueeze(0)
                target_tokens.append(next_tokens)
            else:
                if next_tokens.dim() == 1:
                    next_tokens = next_tokens.unsqueeze(0)
                if streamer is not None:
                    streamer.put(next_tokens[0].cpu())

            target_ids = torch.cat(target_tokens, dim=-1)

            if is_segment_boundary:
                current_seg_idx += 1
                if current_seg_idx < total_segments:
                    last_boundary_token = generated_token_id
                    need_wait_for_segment = True
                    source_input_length = sum(source_seg_len[: current_seg_idx + 1])
                    input_length = (source_input_length, target_input_length)
                else:
                    boundary_tensor = torch.tensor(
                        [[generated_token_id]], dtype=torch.long, device=input_ids.device
                    )
                    target_tokens.append(boundary_tensor)
                    input_ids = torch.cat([input_ids, boundary_tensor], dim=-1)
                    target_ids = torch.cat(target_tokens, dim=-1)

            elif is_final_end:
                unfinished_sequences = unfinished_sequences * 0
                this_peer_finished = True

            target_generated_len = target_ids.shape[-1] - 2
            max_new = generation_config.max_new_tokens if generation_config.max_new_tokens is not None else float("inf")
            if target_generated_len >= max_new and not is_segment_boundary:
                unfinished_sequences = unfinished_sequences * 0
                this_peer_finished = True

        self.read_stop_flag.set()
        read_thread.join(timeout=5.0)
        self.segment_source_caches.clear()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=target_ids, scores=scores, logits=raw_logits,
                attentions=decoder_attentions, hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return target_ids

    def _validate_model_kwargs_stream(self, model_kwargs: dict[str, Any]):
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation_stream).parameters)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args and key not in TransformersKwargs.__optional_keys__:
                unused.append(key)

        if unused:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused} "
                "(note: typos in the generate arguments will also show up in this list)"
            )

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length,
    ):
        cache_name = "past_key_values"
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            return
        if generation_config.use_cache is False:
            return
        if not self._supports_default_dynamic_cache():
            return

        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in ALL_STATIC_CACHE_IMPLEMENTATIONS:
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                cache_config = generation_config.cache_config or {}
                if "config" not in cache_config:
                    cache_config["config"] = self.config.get_text_config()
                backend = cache_config.pop("backend", "quanto")
                model_kwargs[cache_name] = QuantizedCache(backend=backend, **cache_config)
            else:
                model_kwargs[cache_name] = DynamicCache()
        else:
            model_kwargs[cache_name] = DynamicCache()

        if model_kwargs.get("source_key_values") is None:
            model_kwargs["source_key_values"] = DynamicCache()

    def _get_initial_cache_position_for_streaming(self, input_length, model_kwargs):
        assert self.source_key_values is not None
        cache_position = torch.arange(
            self.source_key_values.get_seq_length(), input_length[0],
            dtype=torch.int64, device=model_kwargs.get("assistant_token").device,
        )
        model_kwargs["cache_position"] = cache_position
        return model_kwargs
