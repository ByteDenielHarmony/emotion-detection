# Emotion detection and intensity estimation

Emotion detection and intensity estimation using SemEval-2025 Task 11 data.
The main experiments focus on English Tracks A and B, using TF-IDF, logistic
regression, and nearest-neighbor features. A separate planned workstream
provides one reference baseline model for each of Tracks A, B, and C, including
multilingual settings, following the original task paper.

## Prerequisites

- Git.
- Python **3.12**. Other Python minor versions are not currently supported.
- Either [uv](https://docs.astral.sh/uv/getting-started/installation/), which can
  download Python automatically, or an existing Python 3.12 installation with
  `venv` and pip.

Clone the repository and run all subsequent commands from its root:

```bash
git clone https://github.com/ByteDenielHarmony/emotion-detection.git
cd emotion-detection
```

## Set up the environment

Choose one of the following methods, starting with a fresh virtual environment.
Both install runtime dependencies and the development tools pytest and Ruff.

### Option A: uv (recommended)

```bash
uv sync --locked
```

This creates `.venv`, uses Python 3.12 from `.python-version`, and installs the
versions recorded in `uv.lock`. `--locked` fails if the lockfile does not match
`pyproject.toml`, rather than silently changing it. Shell activation is not
needed when using `uv run`.

### Option B: Python venv and pip (no uv required)

The committed `requirements.txt` is exported from `uv.lock`, with exact versions
and package hashes. You do not need uv to install it.

Linux/macOS (bash or zsh):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

Windows (Command Prompt):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

Keep the environment activated while using the non-uv commands below. Run
`deactivate` when finished. Unlike uv, pip does not read `.python-version`, so
creating the environment explicitly with Python 3.12 matters.
The editable install makes `emotion_detection` importable. Its build backend is
included in the locked requirements; `--no-build-isolation` avoids downloading
unlocked build dependencies. uv installs the project automatically during sync.

PyTorch is included. NVIDIA training additionally requires a compatible NVIDIA
driver; the locked PyTorch build bundles its CUDA runtime dependencies. CPU
execution remains supported. To check whether CUDA is available:

```bash
uv run --locked python -c "import torch; print(torch.cuda.is_available())"
```

Without uv, run the same command without the `uv run --locked` prefix in the
activated virtual environment.

## Check the installation

With uv:

```bash
uv run --locked python -m pytest
uv run --locked python -m ruff check src tests
```

With the pip virtual environment activated:

```bash
python -m pytest
python -m ruff check src tests
```

Tests use synthetic data and do not require external datasets.

## Track A reference baseline

The model API accepts **already-loaded pandas DataFrames** using the official
Track A columns: `id`, `text`, and binary emotion columns. English uses `anger`,
`fear`, `joy`, `sadness`, and `surprise`; some other languages also use `disgust`.
Labels are aligned by name, not column position. Raw text is not lowercased.
File loading and dataset preparation are deliberately separate from the model.

Given `train_data` and `validation_data` supplied by your data pipeline:

```python
from emotion_detection.reference_baselines import TrackAClassifier

model = TrackAClassifier(
    model_name="FacebookAI/xlm-roberta-large",
    # Use "FacebookAI/xlm-roberta-base" here for a smaller checkpoint.
    device="auto",  # "cuda" requires a GPU; "cpu" forces CPU execution.
)
model.fit(train_data, validation_data)

predictions = model.predict(validation_data)  # id + named binary predictions
probabilities = model.predict_proba(validation_data)
scores = model.evaluate(validation_data)

checkpoint = model.run_dir / "checkpoint"
restored = TrackAClassifier.load(checkpoint)
```

The initial fit downloads the selected pretrained model and tokenizer unless
already cached. Pass `local_files_only=True` to forbid downloads, or supply a
local checkpoint directory as `model_name`. Known base/large revisions are pinned;
an explicit `revision=` overrides the pin. Other Hub checkpoints record their
resolved revision in the run metadata. Reloading a saved checkpoint is local.

The label set defaults to the known emotion columns present in training data.
Use `label_names=(...)` to specify a subset or order explicitly, including when
an upstream table contains an unused all-null emotion column. Missing labels are
never converted to zeros. Train and validation IDs and lowercase text keys must
not overlap; cleaning and split construction remain the caller's responsibility.

Defaults follow the related [BRIGHTER training implementation](https://github.com/emotion-analysis-project/BRIGHTER/blob/08d4a59ad950ad78265322cf3d3dbf41a5888f90/regular_lms_track_ab.py):
two epochs, learning rate `1e-5`, 256 tokens, weighted binary cross-entropy, and
threshold `0.5`. For limited VRAM, `fit` defaults to microbatch 2, gradient
accumulation 4, and gradient checkpointing. CUDA uses BF16 when supported,
otherwise FP16; CPU uses FP32. These settings need a real GPU smoke test before
assuming XLM-R-large fits in 16 GB. Reduce `batch_size` and increase
`gradient_accumulation_steps` if needed. Training options are method arguments,
not external configuration files.

Each fit creates a unique run directory, or accepts a nonexisting `run_id`:

```text
output/runs/<run-id>/
├── checkpoint/     Best validation model, tokenizer, label mapping, and metadata
├── training/       Intermediate Trainer checkpoints, including optimizer state
├── metadata.json   Model revision, settings, versions, and input fingerprints
└── metrics.json    Best model's development-set evaluation
```

Optional `data_metadata={...}` passed to `fit` records dataset provenance supplied
by the caller. The final checkpoint is sufficient for inference; resumable
Trainer checkpoints require more disk space. Existing runs are never overwritten.

This implements the RoBERTa-style reference described in
[the task overview, §4.3](https://aclanthology.org/2025.semeval-1.327/), not a claim
to reproduce its exact scores. Checkpoint choice, microbatching, and selection on
development data are recorded explicitly. Reference Tracks B and C are not yet
implemented; tokenization and transformer utilities are separated for reuse.

## Dataset preparation

TODO.

## Update dependencies

`pyproject.toml` declares dependencies; `uv.lock` records their resolved versions.
Maintainers should use uv to update them, then regenerate the pip export:

```bash
uv add <package>
# For a development-only dependency, use: uv add --dev <package>
uv export --locked --format requirements.txt --no-emit-project --output-file requirements.txt
```

Commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together. Do not edit
`requirements.txt` manually or replace it with `pip freeze` output. The export
includes the default development dependencies, matching `uv sync --locked`.

For reproducible results, use the same Git revision, Python version, locked
dependencies, and raw dataset files. The lockfile alone does not pin the data.

## Repository layout and generated files

| Path | Purpose |
| --- | --- |
| `src/emotion_detection/` | Dataset, feature, model, evaluation, and experiment modules |
| `src/emotion_detection/reference_baselines/` | Modular transformer reference baselines |
| `tests/` | Automated tests and small synthetic fixtures |
| `data/raw/`, `data/processed/` | Original and prepared datasets |
| `output/analysis/`, `output/runs/` | Generated analyses and experiment artifacts |
| `reports/` | LaTeX report sources and references |
| `reports/results/` | Selected result summaries retained in Git |

`data/` and `output/` ignore all contents except `.gitkeep` placeholders. Keep
large datasets and trained models out of tracked directories. The virtual
environment and LaTeX build artifacts are also ignored.
