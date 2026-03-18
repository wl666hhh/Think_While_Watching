import gc
import json
import logging
import os
import re
import signal
import sys
from argparse import ArgumentParser, Namespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from models.Qwen3_VL.qwen3_vl_streaming import (
    Qwen3VLForConditionalGeneration_stream,
    Qwen3VLProcessor_stream,
)
from dataloader import StreamingDataCollator

logger = logging.getLogger(__name__)

def load_config_as_args(json_path: str) -> Namespace:
    with open(json_path, 'r', encoding='utf-8') as f:
        return Namespace(**json.load(f))

def setup_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _split_by_boundaries(text: str, EOQ: str, EOT: str):
    if not text:
        return []
    pattern = f"({re.escape(EOQ)}|{re.escape(EOT)})"
    parts = re.split(pattern, text)
    segs, buf = [], ""
    for p in parts:
        if p == EOQ:
            segs.append({"text": buf, "boundary": "EOQ"})
            buf = ""
        elif p == EOT:
            segs.append({"text": buf, "boundary": "EOT"})
            buf = ""
        else:
            buf += p
    if buf.strip():
        segs.append({"text": buf, "boundary": None})
    return segs

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="configs/inference_config.json")
    parser.add_argument("--model_name", type=str, default=None)
    cli_args, _ = parser.parse_known_args()

    args = load_config_as_args(cli_args.config_path)
    if cli_args.data_path:
        args.data_path = cli_args.data_path
    setup_seed(0)

    model_name = cli_args.model_name or os.environ.get("EVAL_MODEL_NAME", "model")
    write_back_dir = os.environ.get("EVAL_WRITE_BACK_DIR", "")
    skip_existing = os.environ.get("EVAL_SKIP_EXISTING", "1") != "0"

    base_model_path = args.base_model_path
    checkpoint_path = args.MLLM_path

    config = AutoConfig.from_pretrained(checkpoint_path)
    config._attn_implementation = "sdpa"

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, padding_side='right')
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='right', config=config)

    processor = Qwen3VLProcessor_stream.from_pretrained(
        base_model_path,
        EOQ=getattr(args, 'EOQ', '<EOQ>'),
        EOT=getattr(args, 'EOT', '<EOT>'),
    )
    processor.tokenizer = tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration_stream.from_pretrained(
        checkpoint_path,
        ignore_mismatched_sizes=True,
        config=config,
        attn_implementation="sdpa",
    )

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        model.resize_token_embeddings(tokenizer_vocab_size)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    data_collator = StreamingDataCollator(
        data_path=args.data_path,
        tokenizer=tokenizer,
        target_start_pos=args.target_start_pos,
        Instruct=args.Instruct,
        user_Instruct=args.user_Instruct,
        assistant_Instruct=args.assistant_Instruct,
        end_Instruct=args.end_Instruct,
        processor=processor,
        vision_start_id=args.vision_start_id,
        vision_end_id=args.vision_end_id,
        video_pad_id=args.video_pad_id,
        EOQ_id=args.EOQ_id,
        EOT_id=args.EOT_id,
        seg_think_Instruct=args.seg_think_Instruct,
        q_think_Instruct=args.q_think_Instruct,
    )

    dataset = data_collator.dataset_loader()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator.collate_fn_inference)
    total_samples = len(dataset)

    device = accelerator.device
    model = model.to(device)
    model.eval()
    stream_model = model

    skipped_count = 0
    processed_count = 0
    initial_completed = 0
    if skip_existing and write_back_dir and accelerator.is_main_process:
        for i in range(total_samples):
            sample_path = os.path.join(write_back_dir, f"data_{i:06d}.json")
            if os.path.isfile(sample_path):
                try:
                    with open(sample_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if existing.get("metadata", {}).get("pred", {}).get(model_name) is not None:
                        initial_completed += 1
                except Exception:
                    pass

    SAMPLE_TIMEOUT = 600
    pbar = tqdm(total=total_samples, initial=initial_completed, disable=not accelerator.is_main_process, desc="Inference")
    dataloader_iter = iter(dataloader)

    while True:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"Dataloader error, skipping: {e}")
            pbar.update(accelerator.num_processes)
            continue

        if skip_existing and write_back_dir:
            meta0 = None
            if isinstance(batch.get("metadata_list"), list) and batch["metadata_list"]:
                meta0 = batch["metadata_list"][0]
            if isinstance(meta0, dict):
                idx = _safe_int(meta0.get("index"))
                if idx is not None:
                    sample_path = os.path.join(write_back_dir, f"data_{idx:06d}.json")
                    if os.path.isfile(sample_path):
                        try:
                            with open(sample_path, "r", encoding="utf-8") as f:
                                existing = json.load(f)
                            if existing.get("metadata", {}).get("pred", {}).get(model_name) is not None:
                                skipped_count += 1
                                pbar.update(accelerator.num_processes)
                                continue
                        except Exception:
                            pass

        inputs = batch["source_tokens"][0]
        inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        assistant_token = batch.get("assistant_token")
        if isinstance(assistant_token, torch.Tensor):
            assistant_token = assistant_token.to(accelerator.device)

        unwrapped_model = stream_model
        output_sequences = None

        for _retry in range(3):
            try:
                if accelerator.is_main_process:
                    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("Timeout")))
                    signal.alarm(SAMPLE_TIMEOUT)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output_sequences = unwrapped_model.generate(
                            **inputs,
                            tokenizer=tokenizer,
                            end_Instruct="<|im_end|>\n",
                            assistant_token=assistant_token,
                            _lengths=batch.get("_lengths"),
                            _lengths_index=batch.get("_lengths_index"),
                            target_start_pos=batch.get("target_start_pos", 0),
                            max_new_tokens=getattr(args, 'max_new_tokens', 16384),
                            seg_think_Instruct_token=batch.get("seg_think_Instruct_token"),
                            q_think_Instruct_token=batch.get("q_think_Instruct_token"),
                        )

                if accelerator.is_main_process:
                    signal.alarm(0)
                break

            except torch.cuda.OutOfMemoryError:
                if accelerator.is_main_process:
                    signal.alarm(0)
                logger.warning(f"OOM, retry {_retry + 1}/3")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                if accelerator.is_main_process:
                    signal.alarm(0)
                logger.warning(f"Error: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                break

        if output_sequences is None:
            pbar.update(accelerator.num_processes)
            continue

        assistant_token_len = assistant_token.shape[0] if assistant_token is not None else 0
        generated_ids = [out[assistant_token_len:] for out in output_sequences]
        output_clean = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_raw = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        if write_back_dir:
            meta0 = batch["metadata_list"][0] if batch.get("metadata_list") else {}
            idx = _safe_int(meta0.get("index") if isinstance(meta0, dict) else None)
            if idx is not None:
                sample_path = os.path.join(write_back_dir, f"data_{idx:06d}.json")
                sample = None
                if os.path.isfile(sample_path):
                    try:
                        with open(sample_path, "r", encoding="utf-8") as f:
                            sample = json.load(f)
                    except Exception:
                        pass

                if isinstance(sample, dict):
                    md = sample.setdefault("metadata", {})
                    pred = md.setdefault("pred", {})

                    EOQ = getattr(args, "EOQ", "<EOQ>")
                    EOT = getattr(args, "EOT", "<EOT>")
                    segs = _split_by_boundaries(output_raw[0], EOQ=EOQ, EOT=EOT)

                    payload = {
                        "raw_text": output_clean[0],
                        "raw_text_with_boundaries": output_raw[0],
                        "segments": segs,
                    }

                    pred[model_name] = payload
                    with open(sample_path, "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2, ensure_ascii=False)

                    processed_count += 1
                    pbar.update(accelerator.num_processes)

    pbar.close()

if __name__ == "__main__":
    main()
