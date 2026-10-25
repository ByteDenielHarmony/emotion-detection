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

## Command-line application

After setup, the same app is available through either entry point:

```bash
uv run --locked emotion --help
uv run --locked python -m emotion_detection --help
```

Without uv, use `emotion` or `python -m emotion_detection` from the activated
environment. Run commands from the repository root, or pass `--project-root PATH`
before the command or among its options. Use `--verbose` for error tracebacks.
Help and version commands do not import PyTorch, download files, or initialize a
model.

### Environment checks

```bash
uv run --locked emotion doctor --device cuda
```

This reports package versions, CUDA availability, GPU memory, and BF16 support.
Requesting unavailable CUDA is an error; `--device auto` allows CPU fallback.
Training also checks the requested device, so running `doctor` first is optional.

### Prepare and check data

```bash
uv run --locked emotion data languages --track a
uv run --locked emotion data prepare --dataset reference --track all --language eng --non-interactive
uv run --locked emotion data prepare --dataset reference --track a --language eng --non-interactive
uv run --locked emotion data prepare --dataset reference --track b --language eng --non-interactive
uv run --locked emotion data prepare --dataset reference --track c --language eng --non-interactive
uv run --locked emotion data check --dataset reference --track a --language eng
```

Preparation delegates acquisition, verification, and persistence to the dataset
package. It may download missing raw files and regenerate prepared files and
audits, but never overwrites existing raw files. `--non-interactive` disables
manual-recovery prompts. Reference Track A supports 28 registered languages;
`data languages` lists them without acquiring or preparing data. Only the selected
language is prepared, not all languages automatically. Data preparation supports
A/B/C according to the registered manifests and C source protocol.
`--track all` prepares A, then B, then C for one language without training; the
preparation default remains A. All selected manifests and the C source protocol
are checked before acquisition or writes. `--protocol` applies only to C within
the batch. Unsupported combinations fail rather than being skipped; a later
failure preserves completed outputs. Joint A+B remains a separate command below.
`--release` selects a registered pinned dataset release. The older
dataset CLI's `--list-languages` option remains available.

`data check` is strictly read-only: it validates the release, source manifest,
canonical data fingerprints, row counts, annotations, and split separation. It
fails with preparation instructions if files or provenance are missing or stale;
it never repairs or downloads anything. Existing reference outputs prepared before
the machine-readable `metadata.csv` was introduced need one explicit re-preparation.
Readiness checks currently cover reference datasets only.

The separate English joint A+B pipeline is still available unchanged:

```bash
uv run --locked emotion data prepare --dataset joint --non-interactive
```

### Train the reference models

```bash
# Default: independent A, then B, then C models for English.
uv run --locked emotion train reference --device cuda

# Train only one track; for C, --language is the target language.
uv run --locked emotion train reference --track a --language eng --device cuda
uv run --locked emotion train reference --track b --language eng --device cuda
uv run --locked emotion train reference --track c --language eng --device cuda
```

**The training default is now `--track all`, not Track A alone.** This selects the
three tracks for one language, not all registered languages. All requested data
and protocol settings are checked before any model is loaded. Unsupported
combinations fail upfront: for example, Romanian lacks B training data at this
release, and Sundanese has no permitted C sources under the default protocol.
Each track initializes fresh weights from the same configured pretrained source;
A/B fine-tuning is never automatically reused for C. Known project checkpoints
containing target-language supervision are rejected as C warm starts; callers
must also verify the provenance of any externally fine-tuned checkpoint.

Models train sequentially, releasing model/trainer references and cached GPU
memory between runs. C pools other-language data (eleven source languages for an
English target), so it is considerably larger than the English A/B tasks. Each
track retains its own checkpoints, so a full suite also needs more disk space.

`--monitor auto` is the default: A/C maximize macro-F1, B maximizes mean Pearson.
`--monitor loss` minimizes each track's own loss. Explicit `macro_f1` is valid only
for A/C, and `pearson` only for B; incompatible suite overrides fail before
training. `--threshold` applies to A/C only; B predicts integer classes by argmax.
`--protocol` is passed only to C (including within an all-track run).

Training uses the read-only prepared-data loader. It never prepares datasets
implicitly. To explicitly run preparation before training instead:

```bash
uv run --locked emotion train reference --language eng --device cuda --prepare-data --non-interactive
```

Use `--model-name FacebookAI/xlm-roberta-base` for a smaller model, or a local
checkpoint path. Other options include `--revision`, `--epochs`, `--learning-rate`,
`--batch-size`, `--gradient-accumulation-steps`, `--max-length`, `--seed`,
`--threshold`, `--mixed-precision`, `--no-gradient-checkpointing`, and `--run-id`.
Run `emotion train reference --help` for defaults and all options.
`--local-files-only` forbids pretrained model/tokenizer downloads; it does not
prevent dataset downloads explicitly requested by `--prepare-data`.

Runs are saved under `output/runs/<run-id>/`, and existing runs are never
overwritten. Training, early stopping, and checkpoint selection use development
data only. **After the selected best checkpoint has been exported, the CLI
evaluates it once on the held-out test set by default.** It prints test loss,
task-appropriate metrics and confusion matrices. These results never
change the selected checkpoint or threshold.

Pass `--skip-test` for development-only runs. Repeatedly inspecting test scores
while deciding model settings compromises the holdout; use development results
for those choices and reserve test results for the final comparison. Test artifacts
are separate: `test_metrics.json`, `test_report.json`, and `test_predictions.csv`
(IDs and binary A/C labels or integer B intensities in original order). Existing results are never
overwritten. If final testing fails, the trained checkpoint and development
artifacts remain saved, and the CLI exits with an error identifying the run.

Test inference follows the run's BF16/FP16 setting on CUDA; CPU uses FP32, and the
actual inference precision is recorded in the test report. The report also records
the selected epoch, frozen threshold, and test-data fingerprint. Development
results in `metrics.json` and `validation_report.json` are not overwritten.
Standalone evaluation/prediction CLI commands are not implemented yet.

### Longer training with early stopping

The default remains two epochs with no early stopping. Task-appropriate selection
metrics are resolved automatically; no `--monitor` flag is needed. To set a larger epoch
limit and stop when development macro-F1 stops improving:

```bash
uv run --locked emotion train reference --track a --language eng --device cuda \
  --epochs 20 --early-stopping-patience 3 \
  --early-stopping-min-delta 0.001
```

This example uses bash line continuations; the same options can be entered on one
line in other shells. A/C select higher macro-F1; B selects higher Pearson;
`--monitor loss` selects lower loss. Checkpoint selection and early stopping use
the same metric. Supplying `--early-stopping-patience` enables stopping after that
many consecutive evaluation epochs without sufficient improvement.
`--early-stopping-min-delta` is the absolute improvement required to reset patience
(default zero), and requires patience to be enabled. It does not change checkpoint
ranking: even a smaller improvement can become the best saved checkpoint.

Every epoch prints development loss, macro/micro-F1, per-emotion precision,
recall, F1 and support, followed by one binary confusion matrix per emotion.
Matrices use actual absent/present as rows and predicted absent/present as columns:
`[[TN, FP], [FN, TP]]`. Reports reuse validation predictions, without another
inference pass or any test evaluation. A final report is labeled separately as
the selected checkpoint's development evaluation, followed by the held-out test
report unless `--skip-test` was supplied.
For B, reports instead show Pearson/MAE/accuracy and 4x4 intensity confusion
matrices with class order 0,1,2,3. Undefined Pearson correlations (constant targets
or predictions, or fewer than two observations) are flagged and contribute zero
to the fixed-label mean, matching the related BRIGHTER baseline convention.

The trainer saves at every epoch and rotates old checkpoints, retaining at most
two intermediate checkpoints while protecting the best and latest. An extra
recent checkpoint may remain when best and latest coincide. Rotation only affects
the new run's `training/` directory, never other runs or existing final checkpoints.
Large optimizer-state snapshots take substantial disk space; this is not an
unlimited checkpoint archive. The selected best model is still exported to
`checkpoint/` when training finishes.

`epoch_history.jsonl` retains metrics and confusion matrices for every completed
epoch, even after its weights are removed. `validation_report.json` contains the
selected model's report. Run metadata records the monitoring settings, completed
epochs, stopping reason, selected epoch, and selected model's loss/F1. The legacy
`best_macro_f1` field records the highest observed epoch F1; when monitoring loss,
this need not equal `selected_checkpoint_macro_f1`.

Longer/early-stopped runs are separate experiments, not exact reproductions of
the original fixed two-epoch recipe. The previous runs are not modified.

## Reference model APIs

`ReferenceTransformer` shares tokenization, training, inference, early stopping,
checkpointing, and persistence. The small `TrackAClassifier`, `TrackBClassifier`,
and `TrackCClassifier` wrappers select binary or intensity task strategies.
The existing Track A API and legacy saved checkpoints remain supported.

| Track | Head and objective | Selection partition | Primary metric |
| --- | --- | --- | --- |
| A | Binary emotion logits; weighted BCE | `dev` | Macro-F1 |
| B | Four logits per emotion; cross-entropy | `dev` | Mean Pearson |
| C | Binary model logits; masked weighted BCE | **`source_dev`** | Masked macro-F1 |

Track C uses `CrosslingualReferenceDataset.train` and `.source_dev` for fitting.
`.target_dev` is not used for model selection or automatically scored. Missing
annotations stay nullable in DataFrames; tensor placeholders carry false masks
and contribute nothing to loss, class weights, or metrics. Training identities
use `(source_language, id)`. C checkpoints store separate model emotion names
(six outputs) and target-native evaluation names; prediction CSVs contain only
the target's categories. Unobserved labels are marked undefined and excluded from
masked source-dev macro-F1, rather than being treated as negative annotations.

For B, `predict` returns ordinal classes 0-3 and `predict_proba` returns named
`<emotion>_<level>` probability columns. `load_reference_model(checkpoint_path)`
dispatches A/B/C automatically; `TrackAClassifier.load(...)` remains compatible.

### Track A example

The model API accepts **already-loaded pandas DataFrames** using the official
Track A columns: `id`, `text`, and binary emotion columns. English uses `anger`,
`fear`, `joy`, `sadness`, and `surprise`. Afrikaans uses five labels including
`disgust` but excluding `surprise`; the other registered languages use six labels.
Labels are aligned by name, not column position. Raw text is not lowercased.
File loading and dataset preparation are deliberately separate from the model.

After explicit preparation, use the read-only loader to supply model inputs:

```python
from emotion_detection.datasets import load_prepared_reference_dataset
from emotion_detection.reference_baselines import TrackAClassifier

data = load_prepared_reference_dataset(track="a", language="eng")
model = TrackAClassifier(
    model_name="FacebookAI/xlm-roberta-large",
    # Use "FacebookAI/xlm-roberta-base" here for a smaller checkpoint.
    device="auto",  # "cuda" requires a GPU; "cpu" forces CPU execution.
    label_names=data.label_names,
)
model.fit(data.train, data.dev, data_metadata=data.metadata)

predictions = model.predict(data.dev)  # id + named binary predictions
probabilities = model.predict_proba(data.dev)
scores = model.evaluate(data.dev)

checkpoint = model.run_dir / "checkpoint"
restored = TrackAClassifier.load(checkpoint)
```

`fit` itself remains development-only; automatic final testing belongs to the CLI
experiment runner, not the model. For an existing frozen checkpoint, explicitly
evaluate it through the same helper (once, without overwriting test artifacts):

```python
from emotion_detection.evaluation.reference import evaluate_reference_test

evaluate_reference_test(restored, data.test, batch_size=2)
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
├── checkpoint/             Best validation model, tokenizer, labels, and metadata
├── training/               Up to two rotating Trainer checkpoints with optimizer state
├── epoch_history.jsonl     All epoch development metrics and confusion matrices
├── validation_report.json  Selected checkpoint's development report
├── metadata.json           Settings, versions, provenance, and stopping summary
├── metrics.json            Training statistics and selected model's dev evaluation
├── test_metrics.json       Held-out test scalar metrics (CLI, unless skipped)
├── test_report.json        Test provenance, metrics, support, and confusion matrices
└── test_predictions.csv    Original test IDs and selected model's task predictions
```

Single-track commands keep this layout. An all-track run groups three such runs:

```text
output/runs/<suite-id>/
├── summary.json
├── a/  # checkpoint/, training/, metrics, reports, predictions
├── b/
└── c/
```

The suite summary records track-specific metrics and statuses, not an average
of F1 and Pearson. Failures stop the suite while preserving completed tracks and
marking pending tracks. `--skip-test` applies to all selected tracks. A suite
`--run-id` names the group; a single-track `--run-id` names the individual run.

Optional `data_metadata={...}` passed to `fit` records dataset provenance supplied
by the caller. The final checkpoint is sufficient for inference; resumable
Trainer checkpoints require more disk space. Existing runs are never overwritten.

This implements the RoBERTa-style reference described in
[the task overview, §4.3](https://aclanthology.org/2025.semeval-1.327/), not a claim
to reproduce its exact scores. Checkpoint choice, microbatching, and selection on
development data are recorded explicitly. B's four-class formulation and C's
source selection/masking follow the related upstream setup with documented
preparation deviations, rather than claiming exact reproduction of every score.

## Dataset preparation

Reference acquisition/preparation and read-only loading are separate APIs:
`prepare_reference_dataset(...)` writes artifacts;
`load_prepared_reference_dataset(...)` only reads and validates them.
The existing `load_reference_dataset(...)` remains an acquisition/preparation
compatibility wrapper; it is **not** a read-only loader.

Reference data is independent of the joint English A+B dataset. Prepared reference
splits are stored in `data/processed/reference_baselines/track_a/<language>/`, and
CSV audits in `output/analysis/reference_baselines/track_a/<language>/`.
`metadata.csv` stores typed metadata in JSON-encoded CSV cells for read-only
loading; `summary.csv` remains the flattened human-readable report. Raw inputs
and the joint pipeline's existing behavior are unchanged.

## Build the LaTeX report

The report starts from Overleaf's [one-column IEEE journal template](https://www.overleaf.com/latex/templates/one-column-ieee-journal-article/ykctxqgxptrs).
Its entry point is `reports/results/bare_jrnl.tex`; accompanying example
bibliographies are under `reports/results/bibtex/`. Keep template license notices
when adapting the sources, and include an AI-usage appendix in the report.

Use [Tectonic **0.17.0**](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0),
installed locally under `.tools/tectonic/`. No systemwide TeX installation or
Python environment is needed. Download the release matching your OS and CPU;
Windows users should use `tectonic.exe` in the build command below.

For **Linux x86_64**, the following bash commands require `curl`, `tar`, and
`sha256sum`. Run from the repository root:

```bash
mkdir -p .tools/tectonic
curl --fail --location \
  'https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz' \
  --output .tools/tectonic/tectonic.tar.gz
sha256sum --check <<'EOF'
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7  .tools/tectonic/tectonic.tar.gz
EOF
```

Continue only if verification reports `OK`. The musl binary is statically linked;
it does not require a separate musl installation.

```bash
tar -xzf .tools/tectonic/tectonic.tar.gz -C .tools/tectonic
.tools/tectonic/tectonic --version
```

Compile from the source directory so relative bibliography and figure paths work:

```bash
cd reports/results
mkdir -p ../build
../../.tools/tectonic/tectonic --keep-logs --outdir ../build bare_jrnl.tex
cd ../..
```

The PDF and log are `reports/build/bare_jrnl.pdf` and
`reports/build/bare_jrnl.log`. Tectonic handles compilation passes and BibTeX when
enabled in the source; the unchanged template uses an inline bibliography.
The first build downloads TeX support files, including `IEEEtran.cls`, into
Tectonic's user cache. Later builds can use `--only-cached` to forbid downloads,
provided all required files are already cached.

**Current limitations:** the unchanged template compiles, but Tectonic substitutes
Latin Modern for IEEEtran's Times fonts and emits fontconfig warnings on the
tested Arch setup. The font setup still needs resolving for faithful template
typography. The executable is version-pinned, but the default TeX bundle is not;
a fully pinned toolchain also needs a compatible bundle and checksum, supplied
via `--bundle PATH`. Do not assume byte-identical PDFs or equivalence to
Overleaf's pdfLaTeX output.

Keep sources, bibliography files, and source figures in Git. `.tools/`,
`reports/build/`, and LaTeX auxiliary outputs are ignored; do not commit downloaded
executables, caches, or generated report PDFs.

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
| `src/emotion_detection/cli.py`, `src/emotion_detection/__main__.py` | Console and module entry points |
| `src/emotion_detection/reference_baselines/` | Modular transformer reference baselines |
| `tests/` | Automated tests and small synthetic fixtures |
| `data/raw/`, `data/processed/` | Original and prepared datasets |
| `output/analysis/`, `output/runs/` | Generated analyses and experiment artifacts |
| `reports/` | LaTeX report sources and references |
| `reports/results/` | Selected result summaries retained in Git |
| `reports/build/` | Ignored compiled report PDFs and build logs |
| `.tools/` | Ignored project-local tools, including Tectonic |

`data/` and `output/` ignore all contents except `.gitkeep` placeholders. Keep
large datasets and trained models out of tracked directories. The virtual
environment and LaTeX build artifacts are also ignored.
