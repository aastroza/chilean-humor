# chilean-humor [Work in progress]

Tracking the history of Chilean humor.

You can visit the previous version of this project (2024) [here](/backup/).

## Data

The dataset includes 135 comedy routines performed at the [Festival de Viña del Mar](https://en.wikipedia.org/wiki/Vi%C3%B1a_del_Mar_International_Song_Festival) between 1960 and 2025. All material is publicly available on YouTube.

The audio was automatically transcribed using the [chirp_2](https://cloud.google.com/speech-to-text?hl=en)  speech-to-text model. The resulting segments were then cleaned, merged, and refined using [gemini-3-flash-preview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash).

## Installation

Using `uv`:

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
