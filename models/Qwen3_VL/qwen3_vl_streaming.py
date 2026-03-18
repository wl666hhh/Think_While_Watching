import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None
    flash_attn_with_kvcache = None

from transformers.cache_utils import Cache, DynamicCache
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.generic import TransformersKwargs, check_model_inputs
from transformers.video_utils import VideoInput
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithPast,
    FlashAttentionKwargs,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    ROPE_INIT_FUNCTIONS,
    deprecate_kwarg,
    dynamic_rope_update,
    eager_attention_forward,
    is_torchdynamo_compiling,
    repeat_kv,
    rotate_half,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from generation.generate import unified_PreTrainedModel
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class Qwen3VLCausalLMOutputWithPast_stream(Qwen3VLCausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
@dataclass
class Qwen3VLModelOutputWithPast_stream(Qwen3VLModelOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
@dataclass
class BaseModelOutputWithPast_stream(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    source_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    causal_mask: Optional[torch.Tensor] = None


class Qwen3VLTextAttention_streaming(Qwen3VLTextAttention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        ReadAction = kwargs.get("ReadAction", False)
        source_key_values = kwargs.get("source_key_values", None)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if ReadAction:
            cache_kwargs_source = {"sin": sin, "cos": cos, "cache_position": None}
            if source_key_values is not None:
                key_states, value_states = source_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs_source
                )
            q_len = query_states.shape[2]
            k_len = key_states.shape[2]
            if HAS_FLASH_ATTN and q_len == k_len and query_states.dtype in [torch.float16, torch.bfloat16]:
                q_flash = query_states.transpose(1, 2)
                k_flash = key_states.transpose(1, 2)
                v_flash = value_states.transpose(1, 2)
                attn_output = flash_attn_func(
                    q_flash, k_flash, v_flash,
                    softmax_scale=self.scaling,
                    causal=True,
                )
                attn_weights = None
            else:
                key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded,
                    attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=True,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_weights = None
        else:
            cache_kwargs_target = {"sin": sin, "cos": cos, "cache_position": None}
            if past_key_values is not None:
                key_states_target, value_states_target = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs_target
                )
            else:
                key_states_target, value_states_target = key_states, value_states
            source_k, source_v = None, None
            source_len = 0
            if source_key_values is not None and source_key_values.get_seq_length() > 0:
                source_layer_cache = source_key_values[self.layer_idx]
                source_k = source_layer_cache[0]
                source_v = source_layer_cache[1]
                source_len = source_k.shape[2]
            q_len = query_states.shape[2]
            target_cache_len = key_states_target.shape[2]
            use_flash_kvcache = (
                HAS_FLASH_ATTN
                and source_k is not None
                and query_states.dtype in [torch.float16, torch.bfloat16]
                and flash_attn_with_kvcache is not None
            )
            if use_flash_kvcache:
                q_flash = query_states.transpose(1, 2)
                k_cache = torch.cat([source_k, key_states_target], dim=2).transpose(1, 2)
                v_cache = torch.cat([source_v, value_states_target], dim=2).transpose(1, 2)
                if q_len == 1:
                    attn_output = flash_attn_with_kvcache(
                        q_flash, k_cache, v_cache,
                        softmax_scale=self.scaling, causal=False,
                    )
                    attn_weights = None
                else:
                    key_states = torch.cat([source_k, key_states_target], dim=2)
                    value_states = torch.cat([source_v, value_states_target], dim=2)
                    k_len = key_states.shape[2]
                    key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                    value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
                    past_len = k_len - q_len
                    row_indices = torch.arange(q_len, device=query_states.device).view(-1, 1)
                    col_indices = torch.arange(k_len, device=query_states.device).view(1, -1)
                    causal_mask = col_indices > (past_len + row_indices)
                    sdpa_mask = torch.zeros(1, 1, q_len, k_len, device=query_states.device, dtype=query_states.dtype)
                    sdpa_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states, key_states_expanded, value_states_expanded,
                        attn_mask=sdpa_mask, dropout_p=0.0, scale=self.scaling, is_causal=False,
                    )
                    attn_output = attn_output.transpose(1, 2)
                    attn_weights = None
            else:
                if source_k is not None:
                    key_states = torch.cat([source_k, key_states_target], dim=2)
                    value_states = torch.cat([source_v, value_states_target], dim=2)
                else:
                    key_states = key_states_target
                    value_states = value_states_target
                k_len = key_states.shape[2]
                key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
                if q_len == 1:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states, key_states_expanded, value_states_expanded,
                        attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=False,
                    )
                else:
                    past_len = k_len - q_len
                    row_indices = torch.arange(q_len, device=query_states.device).view(-1, 1)
                    col_indices = torch.arange(k_len, device=query_states.device).view(1, -1)
                    causal_mask = col_indices > (past_len + row_indices)
                    sdpa_mask = torch.zeros(1, 1, q_len, k_len, device=query_states.device, dtype=query_states.dtype)
                    sdpa_mask.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states, key_states_expanded, value_states_expanded,
                        attn_mask=sdpa_mask, dropout_p=0.0, scale=self.scaling, is_causal=False,
                    )
                attn_output = attn_output.transpose(1, 2)
                attn_weights = None
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, source_key_values, past_key_values


class Qwen3VLTextRotaryEmbedding_streaming(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLProcessor_stream(Qwen3VLProcessor):
    def __init__(self, *args, **kwargs):
        self.EOQ = kwargs.pop("EOQ", "<EOQ>")
        self.EOT = kwargs.pop("EOT", "<EOT>")
        self.system_prompt = kwargs.pop("system_prompt", "You are a helpful assistant.")
        self.data_root = kwargs.pop("data_root", None)
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'chat_template') or self.chat_template is None:
            if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'chat_template'):
                self.chat_template = self.tokenizer.chat_template
    def process_multimodal_inputs(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> Dict[str, Any]:
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps but fps could not be inferred. "
                            "Defaulting to fps=24."
                        )
                        metadata.fps = 24
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices, metadata.fps, self.video_processor.merge_size,
                    )
                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}",
                            video_placeholder, 1,
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        return {
            "text": text,
            "image_inputs": image_inputs,
            "videos_inputs": videos_inputs,
            "output_kwargs": output_kwargs,
            "return_mm_token_type_ids": return_mm_token_type_ids,
            "return_tensors": return_tensors,
        }
    def initialize_inputs_raw_train(self, messages, metadata, data_root=None, for_inference=False):
        EOQ, EOT = self.EOQ, self.EOT
        data_root = data_root or getattr(self, 'data_root', None)
        assert data_root is not None, "data_root is required"
        frames_root = os.path.join(data_root, metadata.get("frames_path", ""))
        frame_files = metadata.get("frame_files", [])
        seg_info = metadata.get("segment_info", [])
        fps = metadata.get("video_fps", 10.0)
        metadata["fps"] = fps
        all_frames_indices = [int(round(t * fps)) for t in metadata.get("sample_timestamps", [])]
        metadata["frames_indices"] = all_frames_indices
        video_segments, seg_bounds, cur = [], [], 0
        for i, s in enumerate(seg_info):
            n = s["num_frames"]
            seg_frames_indices = all_frames_indices[cur:cur + n]
            video_segments.append({
                "segment_id": i,
                "frames": [os.path.join(frames_root, f) for f in frame_files[cur:cur + n]],
                "frames_indices": seg_frames_indices,
                "num_frames": n,
            })
            seg_bounds.append(s["end"])
            cur += n
        user_content, seg_ptr, consumed_segments = [], 0, []
        for m in messages or []:
            if m.get("from") != "human":
                continue
            t = m.get("timestamps")
            if t is not None:
                while seg_ptr < len(seg_bounds) and seg_bounds[seg_ptr] <= t:
                    seg = video_segments[seg_ptr]
                    user_content.append({"type": "video", "video": seg["frames"], "fps": fps, "segment_id": seg_ptr})
                    consumed_segments.append(seg)
                    seg_ptr += 1
            q = m.get("value", "").replace("<video>\n", "").replace("<video>", "").strip()
            if q:
                user_content.append({"type": "text", "text": f"{q}{EOQ}"})
        assert user_content, f"user_content is empty: index={metadata.get('index')}"
        metadata["user_content_structure"] = [item["type"] for item in user_content]
        metadata["segment_lengths"] = [seg["num_frames"] for seg in consumed_segments]
        metadata["discarded_segment_count"] = len(video_segments) - seg_ptr
        assistant_text = ""
        new_messages = [
            {"role": "system", "content": getattr(self, "system_prompt", "You are a helpful assistant.")},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        text = self.apply_chat_template(new_messages, tokenize=False, add_generation_prompt=False)
        images, videos, video_kwargs = process_vision_info(
            new_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        if videos:
            videos, video_metadatas = list(zip(*videos))
            videos, video_metadatas = list(videos), list(video_metadatas)
            assert len(video_metadatas) == len(consumed_segments)
            for vm, seg in zip(video_metadatas, consumed_segments):
                vm["fps"] = fps
                expected_n = int(seg["num_frames"])
                if expected_n % 2 == 1:
                    expected_n += 1
                fi = list(seg["frames_indices"])
                assert fi, f"frames_indices is empty: index={metadata.get('index')}"
                if len(fi) < expected_n:
                    fi.extend([fi[-1]] * (expected_n - len(fi)))
                elif len(fi) > expected_n:
                    fi = fi[:expected_n]
                vm["frames_indices"] = fi
        else:
            video_metadatas = None
        multimodal_inputs = self.process_multimodal_inputs(
            text=text, images=images, videos=videos,
            video_metadata=video_metadatas, return_tensors="pt", do_resize=False,
            **video_kwargs,
        )
        multimodal_inputs.update({
            "video_metadatas": video_metadatas,
            "video_kwargs": video_kwargs,
            "metadata": metadata,
            "boundary_tokens": {"EOQ": EOQ, "EOT": EOT, "num_segments": seg_ptr},
        })
        return multimodal_inputs
    def tokenize_and_merge(
        self,
        text: List[str],
        image_inputs: Dict[str, Any],
        videos_inputs: Dict[str, Any],
        output_kwargs: Dict[str, Any],
        return_mm_token_type_ids: bool,
        return_tensors: str,
    ) -> BatchFeature:
        text_inputs = self.tokenizer(
            text, padding=True, truncation=True,
            add_special_tokens=False, return_token_type_ids=False,
        )
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )


class Qwen3VLTextDecoderLayer_streaming(Qwen3VLTextDecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention_streaming(config=config, layer_idx=layer_idx)
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, attn_weights, source_key_values, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + attn_outputs
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (source_key_values, past_key_values)
        return outputs


class Qwen3VLTextModel_stream(Qwen3VLTextModel):
    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer_streaming(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding_streaming(config=config)
        self.gradient_checkpointing = False
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        use_legacy_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        ReadAction: Optional[bool] = False,
        source_key_values: Optional[Cache] = None,
        _lengths: Optional[List[dict]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast_stream]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        if ReadAction:
            history_source_length = source_key_values.get_seq_length()
            current_length = history_source_length + inputs_embeds.shape[1]
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask[:, :current_length],
                cache_position=cache_position,
                past_key_values=source_key_values,
                position_ids=text_position_ids,
            )
        else:
            attention_mask = None
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                ReadAction=ReadAction,
                source_key_values=source_key_values,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[-1]
                source_key_values = layer_outputs[-2]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx],
                )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast_stream(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            source_key_values=source_key_values,
            causal_mask=attention_mask,
        )


class Qwen3VLModel_stream(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel_stream._from_config(config.text_config)
        self.rope_deltas = None
        self.max_position_ids = None
        self.cached_pixel_values_videos = None
        self.cached_video_grid_thw = None
        self.current_video_grid_thw = None
        self.processed_video_frames = 0
        self.target_generated_len = 0
    def reset_video_cache(self):
        self.cached_pixel_values_videos = None
        self.cached_video_grid_thw = None
        self.current_video_grid_thw = None
        self.processed_video_frames = 0
        self.max_position_ids = None
        self.rope_deltas = None
        self.target_generated_len = 0
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ReadAction: Optional[bool] = False,
        target_start_pos: Optional[int] = 0,
        _lengths: Optional[List[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast_stream]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        self.source_key_values = source_key_values
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if cache_position is None:
            if ReadAction:
                history_source_length = source_key_values.get_seq_length()
                current_length = history_source_length + inputs_embeds.shape[1]
                cache_position = torch.arange(
                    history_source_length, current_length, dtype=torch.long, device=inputs_embeds.device
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
        assert input_ids.shape[1] == cache_position.shape[0], "cache_position must match input length"
        image_mask = None
        video_mask = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None or self.cached_pixel_values_videos is not None:
            if (
                self.cached_pixel_values_videos is None
                or self.cached_video_grid_thw is None
                or self.processed_video_frames == 0
            ):
                self.cached_pixel_values_videos = pixel_values_videos
                self.cached_video_grid_thw = video_grid_thw
                self.processed_video_frames = 0
            n_video_tokens_current = (input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens_current != 0:
                grid_thw = self.cached_video_grid_thw
                tokens_per_frame = (grid_thw[0, 1] * grid_thw[0, 2]) // (self.visual.spatial_merge_size ** 2)
                frame_now = n_video_tokens_current // tokens_per_frame
                if frame_now > 0:
                    patches_per_frame = (grid_thw[0, 1] * grid_thw[0, 2]).item()
                    start_patch = self.processed_video_frames * patches_per_frame
                    end_patch = (self.processed_video_frames + frame_now) * patches_per_frame
                    pixel_values_videos_stream = self.cached_pixel_values_videos[start_patch:end_patch]
                    video_grid_thw_stream = torch.tensor(
                        [[frame_now, grid_thw[0, 1].item(), grid_thw[0, 2].item()]],
                        dtype=grid_thw.dtype, device=grid_thw.device,
                    )
                    self.current_video_grid_thw = video_grid_thw_stream
                    video_embeds, deepstack_video_embeds = self.get_video_features(
                        pixel_values_videos_stream, video_grid_thw_stream
                    )
                    video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                    assert video_embeds.shape[0] == n_video_tokens_current, (
                        f"Video token mismatch: {n_video_tokens_current} vs {video_embeds.shape[0]}"
                    )
                    self.processed_video_frames += frame_now
                    _, video_mask = self.get_placeholder_mask(
                        input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            attn_mask_for_rope = None
            if attention_mask_tensor is not None:
                attn_mask_for_rope = attention_mask_tensor[..., :input_ids.shape[-1]]
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index_streaming(
                    prefill=True,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=self.current_video_grid_thw if self.current_video_grid_thw is not None else video_grid_thw,
                    attention_mask=attn_mask_for_rope,
                    target_start_pos=target_start_pos,
                    ReadAction=ReadAction,
                    device=inputs_embeds.device,
                    _lengths=_lengths,
                )
                self.rope_deltas = rope_deltas
            else:
                position_ids, _ = self.get_rope_index_streaming(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=self.current_video_grid_thw if self.current_video_grid_thw is not None else video_grid_thw,
                    attention_mask=attn_mask_for_rope,
                    prefill=False,
                    target_start_pos=target_start_pos,
                    ReadAction=ReadAction,
                    cache_position=cache_position,
                    device=inputs_embeds.device,
                    _lengths=_lengths,
                )
        assert input_ids.shape[1] == position_ids.shape[2], "position_ids must match input length"
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            source_key_values=source_key_values,
            ReadAction=ReadAction,
            _lengths=_lengths,
            **kwargs,
        )
        return Qwen3VLModelOutputWithPast_stream(
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            source_key_values=outputs.source_key_values,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )
    def _compute_visual_position_ids_for_sample(
        self,
        input_tokens: list,
        image_nums: int,
        video_nums: int,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        image_index: int,
        video_index: int,
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        initial_st_idx: int = 0,
        target_start_pos: int = 0,
        ReadAction: bool = False,
    ) -> tuple[torch.Tensor, int, int]:
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0) else len(input_tokens) + 1
            ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0) else len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
            text_len = ed - st
            current_st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else initial_st_idx
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + current_st_idx)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + current_st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w
        if st < len(input_tokens):
            text_len = len(input_tokens) - st
            if not ReadAction:
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + target_start_pos)
            else:
                current_st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else initial_st_idx
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + current_st_idx)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        return llm_positions, image_index, video_index
    def _compute_simple_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device_to_use = device if device is not None else input_ids.device
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(device_to_use)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=device_to_use)
                .view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=device_to_use, dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas
    def get_rope_index_streaming(
        self,
        prefill: Optional[bool] = True,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_start_pos: Optional[int] = 0,
        ReadAction: Optional[bool] = False,
        device: Optional[torch.device] = None,
        is_training: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        _lengths: Optional[List[dict]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not prefill and not ReadAction:
            batch_size = input_ids.shape[0]
            device_to_use = device if device is not None else input_ids.device
            seq_len = input_ids.shape[1]
            base_position = getattr(self, 'target_generated_len', 0)
            position_ids = torch.arange(seq_len, device=device_to_use).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1) + base_position
            self.target_generated_len = base_position + seq_len
            return position_ids, self.rope_deltas
        has_vision = input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None)
        if not has_vision:
            if ReadAction:
                if self.max_position_ids is not None:
                    base_position = self.max_position_ids.max().item() + 1
                    position_ids = (cache_position.unsqueeze(0).unsqueeze(0) - cache_position[0] + base_position).expand(3, input_ids.shape[0], -1)
                else:
                    position_ids = cache_position.unsqueeze(0).unsqueeze(0).expand(3, input_ids.shape[0], -1)
                self.max_position_ids = position_ids.max(dim=-1, keepdim=True)[0]
                return position_ids, self.rope_deltas
            else:
                if self.max_position_ids is None:
                    return self._compute_simple_position_ids(input_ids, attention_mask, device)
                else:
                    base_position = self.max_position_ids.max().item() + 1
                    seq_len = input_ids.shape[1]
                    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(3, input_ids.shape[0], -1) + base_position
                    self.max_position_ids = position_ids.max(dim=-1, keepdim=True)[0]
                    return position_ids, self.rope_deltas
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        device_to_use = device if device is not None else input_ids.device
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=device_to_use)
        mrope_position_deltas = []
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_seq in enumerate(total_input_ids):
            input_ids_seq = input_ids_seq[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_seq == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_seq[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum().item()
            video_nums = (vision_tokens == video_token_id).sum().item()
            input_tokens = input_ids_seq.tolist()
            initial_st_idx = 0
            if ReadAction and self.max_position_ids is not None:
                initial_st_idx = self.max_position_ids[:, i:i + 1, :].max().item() + 1
            llm_positions, image_index, video_index = self._compute_visual_position_ids_for_sample(
                input_tokens=input_tokens,
                image_nums=image_nums, video_nums=video_nums,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                image_index=image_index, video_index=video_index,
                spatial_merge_size=spatial_merge_size,
                image_token_id=image_token_id, video_token_id=video_token_id,
                initial_st_idx=initial_st_idx,
                target_start_pos=target_start_pos,
                ReadAction=ReadAction,
            )
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=device_to_use).unsqueeze(1)
        if ReadAction:
            self.max_position_ids = position_ids.max(dim=-1, keepdim=True)[0]
            self.rope_deltas = mrope_position_deltas
        return position_ids, mrope_position_deltas
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_image_mask = (inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )).all(-1)
            special_video_mask = (inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )
        return special_image_mask, special_video_mask


class Qwen3VLForConditionalGeneration_stream(unified_PreTrainedModel, Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel_stream(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        source_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ReadAction: Optional[bool] = False,
        target_start_pos: Optional[int] = 0,
        _lengths: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast_stream]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            source_key_values=source_key_values,
            attention_mask=attention_mask,
            ReadAction=ReadAction,
            cache_position=cache_position,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            target_start_pos=target_start_pos,
            _lengths=_lengths,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return Qwen3VLCausalLMOutputWithPast_stream(
            loss=None,
            logits=logits,
            source_key_values=outputs.source_key_values,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
    def prepare_inputs_for_generation_stream(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_length=None,
        is_streaming=True,
        assistant_token=None,
        ReadAction=True,
        **kwargs,
    ):
        assert input_length is not None, "input_length is required for streaming generation"
        model_inputs = kwargs.copy()
        target_start_pos = kwargs.get("target_start_pos", 0)
        if self.source_key_values is not None:
            past_source_length = self.source_key_values.get_seq_length()
        if ReadAction:
            input_ids = input_ids[:, past_source_length:input_length[0]]
        if ReadAction:
            model_inputs.update({
                "input_ids": input_ids,
                "position_ids": None,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "source_key_values": self.source_key_values,
                "ReadAction": ReadAction,
                "target_start_pos": target_start_pos,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "logits_to_keep": kwargs["logits_to_keep"],
            })
        else:
            model_inputs.update({
                "input_ids": input_ids,
                "position_ids": None,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "past_key_values": self.past_key_values,
                "source_key_values": self.source_key_values,
                "ReadAction": ReadAction,
                "target_start_pos": target_start_pos,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "logits_to_keep": kwargs["logits_to_keep"],
            })
        if cache_position[0] != 0 and not ReadAction:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs
