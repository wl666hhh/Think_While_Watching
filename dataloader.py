import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from utils.process_utils import seperate_source_target

logger = logging.getLogger(__name__)

class StreamingDataCollator:

    def __init__(
        self,
        data_path: str,
        processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizer,
        Instruct: str,
        user_Instruct: str,
        assistant_Instruct: str,
        end_Instruct: str,
        target_start_pos: int = 0,
        vision_start_id: int = 151652,
        vision_end_id: int = 151653,
        video_pad_id: int = 151654,
        EOQ_id: int = 151669,
        EOT_id: int = 151670,
        seg_think_Instruct: str = "[SEG THINK]\nFocus: ",
        q_think_Instruct: str = "[Q THINK]\nReasoning: ",
    ):
        self.data_path = data_path
        self.data_root = os.path.dirname(data_path.rstrip('/'))
        self.Instruct = Instruct
        self.user_Instruct = user_Instruct
        self.assistant_Instruct = assistant_Instruct
        self.end_Instruct = end_Instruct
        self.processor = processor
        self.processor.data_root = self.data_root
        self.target_start_pos = target_start_pos
        self.tokenizer = tokenizer

        self.vision_start_id = vision_start_id
        self.vision_end_id = vision_end_id
        self.vision_pad_id = video_pad_id
        self.EOQ_id = EOQ_id
        self.EOT_id = EOT_id

        self.assistant_ids = self.tokenizer(self.assistant_Instruct, add_special_tokens=False)["input_ids"]
        self.assistant_token_tensor = torch.tensor(self.assistant_ids, dtype=torch.long)

        self.seg_think_Instruct = seg_think_Instruct
        self.q_think_Instruct = q_think_Instruct
        self.seg_think_Instruct_ids = self.tokenizer(seg_think_Instruct, add_special_tokens=False)["input_ids"]
        self.q_think_Instruct_ids = self.tokenizer(q_think_Instruct, add_special_tokens=False)["input_ids"]
        self.seg_think_Instruct_token_tensor = torch.tensor(self.seg_think_Instruct_ids, dtype=torch.long)
        self.q_think_Instruct_token_tensor = torch.tensor(self.q_think_Instruct_ids, dtype=torch.long)

    def dataset_loader(self) -> Dataset:
        data_path = Path(self.data_path)
        if not data_path.is_dir():
            raise ValueError(f"Expected a directory, got: {data_path}")

        json_files = sorted(data_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in: {data_path}")

        all_data = []
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                all_data.append(json.load(f))

        return Dataset.from_list(all_data)

    def calculate_lengths(self, source_token, target_token, input_batch_len, **kwargs):
        metadata_list = kwargs.get('metadata_list', [])
        EOQ_id = kwargs.get('EOQ_id', self.EOQ_id)
        EOT_id = kwargs.get('EOT_id', self.EOT_id)
        vision_end_id = kwargs.get('vision_end_id', self.vision_end_id)
        merge_size = 2

        _lengths = []
        attn_mask_index = []

        for index in range(source_token["input_ids"].shape[0]):
            source_id = source_token["input_ids"][index].tolist()
            target_id = target_token["input_ids"][index].tolist()
            metadata = metadata_list[index] if index < len(metadata_list) else {}

            structure = metadata.get("user_content_structure", [])
            segment_lengths = metadata.get("segment_lengths", [])
            vision_blocks_per_video = [math.ceil(n / merge_size) for n in segment_lengths]

            source_seg_lens = []
            current_pos = 0
            video_idx = 0

            for i, item_type in enumerate(structure):
                seg_start = current_pos
                if item_type == "video":
                    num_blocks = vision_blocks_per_video[video_idx] if video_idx < len(vision_blocks_per_video) else 1
                    vision_end_found = 0
                    while current_pos < len(source_id) and vision_end_found < num_blocks:
                        if source_id[current_pos] == vision_end_id:
                            vision_end_found += 1
                        current_pos += 1
                    video_idx += 1
                else:
                    while current_pos < len(source_id) and source_id[current_pos] != EOQ_id:
                        current_pos += 1
                    if current_pos < len(source_id):
                        current_pos += 1

                seg_len = current_pos - seg_start
                if i == 0:
                    source_seg_lens.append(current_pos)
                else:
                    source_seg_lens.append(seg_len)

            if current_pos < len(source_id) and source_seg_lens:
                source_seg_lens[-1] += len(source_id) - current_pos
            elif current_pos < len(source_id):
                source_seg_lens.append(len(source_id) - current_pos)

            source_token_len = sum(source_seg_lens)

            target_seg_lens = []
            current_len = 0
            for tok in target_id:
                current_len += 1
                if tok == EOT_id or tok == EOQ_id:
                    target_seg_lens.append(current_len)
                    current_len = 0
            if current_len > 0 and target_seg_lens:
                target_seg_lens[-1] += current_len
            elif current_len > 0:
                target_seg_lens.append(current_len)

            target_token_len = sum(target_seg_lens)

            assert source_token_len == len(source_id), (
                f"Source length mismatch: {source_token_len} vs {len(source_id)}"
            )
            assert target_token_len == len(target_id), (
                f"Target length mismatch: {target_token_len} vs {len(target_id)}"
            )
            assert len(source_seg_lens) == len(target_seg_lens), (
                f"Segment count mismatch: source={len(source_seg_lens)} vs target={len(target_seg_lens)}"
            )

            _lengths.append({
                'source_token_len': source_token_len,
                'source_seg_len': source_seg_lens,
                'target_token_len': target_token_len,
                'target_seg_len': target_seg_lens,
                'input_token_len': source_token_len + target_token_len,
                'input_batch_len': input_batch_len,
                'segment_types': structure,
            })

            mask_index = torch.zeros((1, input_batch_len))
            mask_index[0, source_token_len:source_token_len + target_token_len] = 2
            first_seg_end = source_seg_lens[0] if source_seg_lens else 0
            if first_seg_end > 0:
                mask_index[0, first_seg_end - 1] = -1
            attn_mask_index.append(mask_index)

        return _lengths, attn_mask_index

    def collate_fn_inference(self, batch_data):
        source_texts = []
        target_texts = []
        input_texts = []
        source_tokens_list = []
        target_tokens_list = []
        _lengths_list = []
        metadata_list = []

        for item in batch_data:
            messages = item["conversations"]
            metadata = item["metadata"]
            metadata_list.append(metadata)

            multimodal_inputs = self.processor.initialize_inputs_raw_train(
                messages=messages, metadata=metadata, data_root=self.data_root, for_inference=True,
            )

            source_text, _ = seperate_source_target(multimodal_inputs["text"])

            structure = metadata.get("user_content_structure", [])
            EOQ = getattr(self.processor, "EOQ", "<EOQ>")
            EOT = getattr(self.processor, "EOT", "<EOT>")
            dummy_body = "".join([EOT if t == "video" else EOQ for t in structure])
            target_text = f"{self.assistant_Instruct}{dummy_body}{self.end_Instruct}"

            source_token = self.processor.tokenize_and_merge(
                text=[source_text],
                image_inputs=multimodal_inputs["image_inputs"],
                videos_inputs=multimodal_inputs["videos_inputs"],
                output_kwargs=multimodal_inputs["output_kwargs"],
                return_mm_token_type_ids=multimodal_inputs["return_mm_token_type_ids"],
                return_tensors=multimodal_inputs["return_tensors"],
            )
            target_token = self.processor.tokenize_and_merge(
                text=[target_text],
                image_inputs={}, videos_inputs={},
                output_kwargs=multimodal_inputs["output_kwargs"],
                return_mm_token_type_ids=multimodal_inputs["return_mm_token_type_ids"],
                return_tensors=multimodal_inputs["return_tensors"],
            )

            source_texts.append(source_text)
            target_texts.append(target_text)
            input_texts.append(multimodal_inputs)
            source_tokens_list.append(source_token)
            target_tokens_list.append(target_token)

        if isinstance(input_texts[0]["text"], list) and len(input_texts[0]["text"]) == 1:
            batch_texts = [item["text"][0] for item in input_texts]
        elif isinstance(input_texts[0]["text"], list):
            batch_texts = [t for item in input_texts for t in item["text"]]
        else:
            batch_texts = [item["text"] for item in input_texts]

        merged_image_inputs = {}
        merged_videos_inputs = {}

        if input_texts[0]["image_inputs"]:
            for key in input_texts[0]["image_inputs"]:
                values = [item["image_inputs"][key] for item in input_texts if key in item["image_inputs"]]
                if values and isinstance(values[0], torch.Tensor):
                    merged_image_inputs[key] = torch.cat(values, dim=0)
                else:
                    merged_image_inputs[key] = [v for sublist in values for v in (sublist if isinstance(sublist, list) else [sublist])]

        if input_texts[0]["videos_inputs"]:
            for key in input_texts[0]["videos_inputs"]:
                values = [item["videos_inputs"][key] for item in input_texts if key in item["videos_inputs"]]
                if values and isinstance(values[0], torch.Tensor):
                    merged_videos_inputs[key] = torch.cat(values, dim=0)
                else:
                    merged_videos_inputs[key] = [v for sublist in values for v in (sublist if isinstance(sublist, list) else [sublist])]

        inputs_tokens_list = self.processor.tokenize_and_merge(
            text=batch_texts,
            image_inputs=merged_image_inputs,
            videos_inputs=merged_videos_inputs,
            output_kwargs=input_texts[0]["output_kwargs"],
            return_mm_token_type_ids=input_texts[0]["return_mm_token_type_ids"],
            return_tensors=input_texts[0]["return_tensors"],
        )

        input_batch_len = inputs_tokens_list.data["input_ids"].shape[1]
        for idx, (source_token, target_token) in enumerate(zip(source_tokens_list, target_tokens_list)):
            _lengths, _ = self.calculate_lengths(
                source_token, target_token, input_batch_len,
                metadata_list=[metadata_list[idx]],
                EOQ_id=self.EOQ_id, EOT_id=self.EOT_id, vision_end_id=self.vision_end_id,
            )
            _lengths_list.append(_lengths[0])

        _lengths_index = torch.tensor(range(len(_lengths_list))).unsqueeze(1)

        return {
            "input_txt": batch_texts,
            "source_txt": source_texts,
            "target_txt": target_texts,
            "source_tokens": source_tokens_list,
            "target_tokens": target_tokens_list,
            "inputs_tokens": inputs_tokens_list,
            "assistant_token": self.assistant_token_tensor.clone(),
            "seg_think_Instruct_token": self.seg_think_Instruct_token_tensor.clone(),
            "q_think_Instruct_token": self.q_think_Instruct_token_tensor.clone(),
            "_lengths": _lengths_list,
            "_lengths_index": _lengths_index,
            "target_start_pos": self.target_start_pos,
            "metadata_list": metadata_list,
        }
