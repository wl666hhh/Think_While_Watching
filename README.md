# Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2603.11896-b31b1b.svg)](https://arxiv.org/abs/2603.11896)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2603.11896)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the **code** for *Think While Watching*, a memory-anchored streaming video reasoning framework that preserves continuous segment-level memory during multi-turn interaction. The model reads video segments and questions incrementally and generates thoughts and answers on-the-fly, enabling real-time online video understanding.


## Project Structure

```
Think_While_Watching/
├── configs/
│   └── inference_config.json      # Inference configuration
├── data/
│   └── sample_data/               # Sample data for testing
│       └── data_000000.json
├── generation/
│   └── generate.py                # Streaming generation logic
├── inference/
│   └── streaming_inference.py     # Main inference entry point
├── models/
│   └── Qwen3_VL/
│       └── qwen3_vl_streaming.py  # Streaming Qwen3-VL model
├── utils/
│   └── process_utils.py           # Text processing utilities
├── dataloader.py                  # Data loading & collation
├── inference.sh                   # bash
├── requirements.txt               # Python dependencies
└── README.md
```

## Requirements

- Python ≥ 3.10
- CUDA ≥ 12.1


Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `flash-attn` needs to download from
https://github.com/Dao-AILab/flash-attention/releases.
## Data Format

Each sample is a JSON file with the following structure:

```json
{
  "video": "/path/to/video.mp4",
  "conversations": [
    {
      "from": "human",
      "timestamps": 5.0,
      "value": "<video>\nWhat is happening?"
    },
    {
      "from": "gpt",
      "value": "A person is walking."
    }
  ],
  "metadata": {
    "dataset": "sample",
    "timestamps": [5.0],
    "video_duration": 15.0,
    "num_frames": 10,
    "frame_files": ["frame_0001.jpg", "..."],
    "sample_timestamps": [1.0, 2.0, "..."],
    "segment_info": [
      {"start": 0.0, "end": 5.0, "num_frames": 5}
    ],
    "frames_path": "raw_picture/0/",
    "index": 0
  }
}
```

A sample data file is provided at `data/sample_data/data_000000.json`.

## Configuration

Edit `configs/inference_config.json`:

| Field | Description |
|-------|-------------|
| `MLLM_path` | Path to the model checkpoint |
| `base_model_path` | Path to the base Qwen3-VL model |
| `data_path` | Directory containing sample JSON files |
| `target_start_pos` | Starting position ID for target tokens (default: 0) |

## Start

### Single GPU

```bash
bash inference.sh --data_path /path/to/data/all_data
```

### Multi-GPU

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash inference.sh --data_path /path/to/data/all_data --multi_gpu
```




## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026thinkwatchingonlinestreaming,
      title={Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models}, 
      author={Lu Wang and Zhuoran Jin and Yupu Hao and Yubo Chen and Kang Liu and Yulong Ao and Jun Zhao},
      year={2026},
      eprint={2603.11896},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.11896}, 
}
```


