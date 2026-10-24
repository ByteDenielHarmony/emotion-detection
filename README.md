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
python -m pip check
```

Windows (Command Prompt):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --require-hashes -r requirements.txt
python -m pip check
```

Keep the environment activated while using the non-uv commands below. Run
`deactivate` when finished. Unlike uv, pip does not read `.python-version`, so
creating the environment explicitly with Python 3.12 matters.

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
| `tests/` | Automated tests and small synthetic fixtures |
| `data/raw/`, `data/processed/` | Original and prepared datasets |
| `output/analysis/`, `output/runs/` | Generated analyses and experiment artifacts |
| `reports/` | LaTeX report sources and references |
| `reports/results/` | Selected result summaries retained in Git |

`data/` and `output/` ignore all contents except `.gitkeep` placeholders. Keep
large datasets and trained models out of tracked directories. The virtual
environment and LaTeX build artifacts are also ignored.
