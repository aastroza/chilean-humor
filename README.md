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

### NVIDIA GPU setup (Jina embeddings)

If you want Jina embeddings to run on GPU, install CUDA-enabled PyTorch wheels
after the base requirements.

Windows (PowerShell):

```powershell
uv venv
.venv\Scripts\activate
uv pip install -r .\requirements.txt
uv pip uninstall torch torchvision
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

macOS/Linux:

```bash
uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt
uv pip uninstall torch torchvision
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The check must print `True` for `torch.cuda.is_available()`.

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

## Topic Modeling With Jina Embeddings

You can precompute Jina embeddings and reuse them across BERTopic runs.

Local provider (`transformers` + optional GPU):

```powershell
.venv\Scripts\python scripts/run_topic_modeling.py `
  --use-jina-embeddings `
  --jina-provider local `
  --jina-model-name jinaai/jina-embeddings-v4 `
  --jina-task text-matching `
  --jina-truncate-dim 128 `
  --jina-device auto
```

Jina API provider (useful when local inference is too slow):

```env
JINA_API_TOKEN=jina_xxx_your_token
```

```powershell
.venv\Scripts\python scripts/run_topic_modeling.py `
  --use-jina-embeddings `
  --jina-provider api `
  --jina-model-name jina-embeddings-v4 `
  --jina-task text-matching
```

Notes:

- Embeddings are cached under `outputs/topic_modeling/embeddings_cache` by default.
- Use `--jina-cache-dir <path>` to keep cache files somewhere else.
- Set `--jina-truncate-dim 0` to disable truncation.
- API mode reads the bearer token from `JINA_API_TOKEN` by default (or another variable via `--jina-api-token-env`).
- API endpoint can be overridden with `--jina-api-url` and timeout with `--jina-api-timeout-seconds`.

Main outputs for topic analysis are written to `outputs/topic_modeling`:

- `tables/segments_topics.csv`: row-level table linked to the original `segments` dataset (original columns + cleaned text + decade + initial/final topic assignment + outlier flags + max topic probability).
- `tables/topic_info.csv`: topic metadata and representative documents.
- `tables/topics_over_time.csv`: topic trends over time.
- `tables/hierarchical_topics.csv`: hierarchy produced by BERTopic.
- `figures/topic_hierarchy.html` and `figures/topic_hierarchy.png`: hierarchical clustering visualization.
- `figures/topics_over_time_top_n.html` and `figures/topics_over_time_top_n.png`: temporal trend visualization.


## Joke Extraction and Mentalizing Analysis Pipeline

This repository implements two complementary structured-output pipelines:

1.  **Joke extraction pipeline**\
2.  **Mentalizing (intentionality) analysis pipeline**

Both pipelines use **structured JSON output enforced by schema
validation**, ensuring deterministic and reproducible results suitable
for downstream analysis.

This work builds upon the cognitive framework proposed by [Dunbar et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26597196/), who demonstrated that verbal jokes rely on recursive mentalizing—the ability to represent nested mindstates such as “A thinks that B thinks…”. Their analysis showed that a conversational exchange minimally requires three levels of intentionality, and that most jokes involve approximately three to five levels, with humor effectiveness peaking within this range. Each additional embedded mindstate increases cognitive load, establishing a natural limit on joke complexity based on human mentalizing capacity.

### Overview

The system processes comedy transcripts in two stages:

    Transcript → Joke Extraction → Individual Jokes → Mentalizing Analysis → Intentionality Tree / Graph

Each stage uses:

-   deterministic preprocessing (heuristics)
-   structured output (JSON schema)
-   post-processing validation (Pydantic models)

### 1. Joke Extraction Pipeline

#### Purpose

Extract clean, standalone jokes from noisy transcripts of comedy
performances.

This stage converts transcript segments into a list of normalized joke
texts.

#### Input

Transcript in structured segment format:

``` json
{
  "segments": [
    { "text": "..." },
    { "text": "..." }
  ]
}
```

Segments may be incomplete or split across boundaries.


#### Output

``` python
List[str]
```

Example:

``` python
[
  "Mi amigo cree que su polola piensa que quiero robarle el perro.",
  "El médico dijo: Toro parado, venid."
]
```


#### Method

The pipeline uses a hybrid approach:

##### Step 1 --- Segment windowing with overlap

Segments are merged into windows with overlap to prevent punchlines from
being cut:

    segment N + segment N+1 overlap → complete joke context


##### Step 2 --- Gemini structured extraction

Gemini is instructed to classify content into:

``` python
kind = "joke" | "non_joke"
```

Non-joke content includes:

-   comedian introductions
-   awards
-   applause segments
-   announcer speech
-   promos or transitions

Only `"joke"` entries are returned.


##### Step 3 --- Cleaning and deduplication

The pipeline:

-   fixes punctuation
-   removes duplicates
-   returns normalized joke text

#### Design goals

-   deterministic output format
-   transcript noise robustness
-   reliable joke boundaries
-   language-agnostic (configured via prompt)


### 2. Mentalizing (Intentionality) Analysis Pipeline

#### Purpose

Compute the **mentalizing complexity** of a joke using explicit
recursive mindstate modeling.

This follows the cognitive framework described in Dunbar et al.

Key concept:

> Intentionality level = depth of recursively embedded mental states

Example:

    comedian intends
      audience understands
        doctor believes
          Toro Sentado is now Toro Parado

Intentionality depth = 4


#### Output

Structured nested tree:

``` json
{
  "root": {
    "holder": "comedian",
    "verb": "intends",
    "content": "...",
    "embeds": {
      "holder": "audience",
      "verb": "understands",
      "content": "...",
      "embeds": {
        ...
      }
    }
  }
}
```

This structure explicitly encodes mental state nesting.


#### Derived outputs

From this tree, the system can compute:

``` python
intentionality_depth = compute_intentionality_depth(root)
```

Example:

    Depth = 5


#### Pipeline Architecture

This pipeline combines deterministic heuristics and structured LLM
reasoning.


##### Step 1 --- Heuristic candidate extraction (deterministic)

The system scans the joke text for mental state indicators:

Examples:

    creo que...
    piensa que...
    quiere...
    espera...

These generate candidate mental states:

``` python
Candidate(
    span="mi amigo cree que...",
    holder_hint="mi amigo",
    verb_hint="believes",
    estimated_depth=2
)
```

This step improves reproducibility and reduces hallucination.


##### Step 2 --- Gemini structured mentalizing reconstruction

Gemini receives:

-   joke text
-   heuristic candidates
-   strict recursive schema

Gemini outputs a minimal nested mental state structure.

The schema enforces explicit embedding:

``` python
NestedMentalState
    embeds NestedMentalState
        embeds NestedMentalState
```


##### Step 3 --- Depth computation

Depth is computed deterministically:

``` python
def compute_intentionality_depth(node):
    if node.embeds is None:
        return 1
    return 1 + compute_intentionality_depth(node.embeds)
```

This produces the intentionality level.


##### Step 4 --- Optional graph visualization

The nested structure can be converted into Graphviz DOT format:

``` dot
ms1 → ms2 → ms3 → ms4 → ms5
```

This allows visualization of the mentalizing chain.


### Why both pipelines are needed

The pipelines solve different but complementary problems:

  Pipeline               Purpose
  ---------------------- -------------------------------------------
  Joke extraction        isolate jokes from transcript noise
  Mentalizing analysis   measure cognitive complexity of each joke

Together, they enable large-scale cognitive analysis of humor.

------------------------------------------------------------------------

### Example end-to-end workflow

``` python
transcript → extract_jokes()

for joke in jokes:
    tree = analyze_joke_mentalizing_tree(joke)
    depth = compute_intentionality_depth(tree.root)
```

Output:

``` python
[
  {
    "joke": "...",
    "intentionality_depth": 5
  }
]
```