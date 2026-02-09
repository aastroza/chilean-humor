# chilean-humor [Work in progress]

Tracking the history of Chilean humor.

You can visit the previous version of this project (2024) [here](/backup/).

**The raw dataset is available on Hugging Face: [astroza/chilean-humor-raw-transcripts](https://huggingface.co/datasets/astroza/chilean-humor-raw-transcripts).**

## Data

The dataset includes 135 comedy routines performed at the [Festival de Viña del Mar](https://en.wikipedia.org/wiki/Vi%C3%B1a_del_Mar_International_Song_Festival) between 1960 and 2025. All material is publicly available on YouTube.

The audio was automatically transcribed using the [`chirp_2`](https://cloud.google.com/speech-to-text?hl=en)  speech-to-text model. The resulting segments were then cleaned, merged, and refined using [`gemini-3-flash-preview`](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash).

## Installation

Using [`uv`](https://docs.astral.sh/uv/):

Windows (PowerShell):

```powershell
uv venv
.venv\Scripts\activate
uv pip install -r .\requirements.txt
```

macOS/Linux:

```bash
uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt
```

## Build HF Dataset from raw data

Generate the two Hugging Face configs from `data/2026`:

```powershell
.venv\Scripts\python scripts/build_hf_dataset.py --input-root data/2026 --output-root hf_dataset
```

This creates:

- `hf_dataset/routines/train.parquet`
- `hf_dataset/routines/train.jsonl`
- `hf_dataset/segments/train.parquet`
- `hf_dataset/segments/train.jsonl`

## Upload Dataset To Hugging Face Hub

1. Set your token in `.env` (or export `HF_TOKEN` in your shell):

```env
HF_TOKEN=hf_xxx_your_token
```

2. Push both configs (`routines` and `segments`) to the same dataset repo:

Windows (PowerShell):

```powershell
python scripts/push_hf_dataset.py --repo-id <your_username>/chilean-humor-raw-transcripts --dataset-root hf_dataset
```

macOS/Linux:

```bash
python scripts/push_hf_dataset.py --repo-id <your_username>/chilean-humor-raw-transcripts --dataset-root hf_dataset
```

Optional: add `--private` if you want the dataset repo to be private when it is first created.
