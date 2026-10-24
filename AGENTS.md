# Contributor and agent guidance

## Scope and current state

- Main experiments focus on English SemEval-2025 Task 11 Tracks A and B.
- The English emotion order is anger, fear, joy, sadness, surprise. Track A
  predicts five binary presence labels; Track B predicts five intensities, 0–3.
- Prioritize necessary Stage 1 work: data preparation, analysis, evaluation,
  the traditional baseline, and reproducibility. Avoid unrequested scope growth.
- Another project goal is to reimplement the original task paper's RoBERTa
  baselines for Tracks A, B, and C, including multilingual and cross-lingual
  settings. Keep these reference baselines separate from the main experiments
  and document checkpoint choices and deviations from the published setup.
  This does not require a single shared checkpoint across languages or exhaustive
  replication of every language/configuration combination or published score.
- Dataset preparation, a modular Track A transformer reference baseline, and
  synthetic tests exist. Reference Tracks B and C are not yet implemented.
  Inspect current code and distinguish implemented, tested, and trained models;
  a passing tiny-model test is not evidence of a pretrained-model benchmark run.

## Environment and dependencies

- Use Python 3.12 for the main project and prefer uv for development. Preserve
  installation through a standard Python virtual environment and pip.
- Add dependencies only when needed. `pyproject.toml` declares dependencies;
  `uv.lock` records resolved versions. After main-project dependency changes,
  regenerate the hash-pinned pip export:

  ```bash
  uv export --locked --format requirements.txt --no-emit-project --output-file requirements.txt
  ```

- Commit dependency changes to `pyproject.toml`, `uv.lock`, and
  `requirements.txt` together. Never edit the export manually or replace it
  with `pip freeze` output.
- If a reference baseline needs different Python or dependency versions, isolate
  and document its environment. Do not silently change the main project's pins
  to accommodate upstream requirements.
- Do not assume Linux is required. Prefer portable paths and Python APIs, and
  identify shell-specific commands in documentation.

## Code organization

- Keep implementation under `src/emotion_detection/`:
  - `datasets/`: acquisition, preparation, dataset dataclasses, and analysis.
  - `features/`: TF-IDF and leakage-safe nearest-neighbor transformations.
  - `models/`: OOP fitting, prediction, persistence, and small shared interfaces.
  - `evaluation/`: reusable metrics.
  - `experiments/`: experiment orchestration and run artifact management.
- Keep CLI argument handling separate from reusable logic. Favor composition,
  existing scikit-learn components, and small interfaces over large hierarchies.
- Use Python modules and CLI commands, not notebooks. Do not introduce an
  experiment configuration framework or extra infrastructure without a request.
- Reference baselines live under `src/emotion_detection/reference_baselines/`.
  Reuse checkpoint/tokenizer utilities and in-memory adapters across tracks,
  while keeping objectives and experiment orchestration track-specific.
- Model APIs receive already-loaded data. Do not add persistent data loading,
  downloading, or preprocessing to the model code; another team member owns that
  integration. Never alter the existing preparation pipeline incidentally.

## Data integrity and main-experiment evaluation

- Never modify raw source files. Preserve provenance and document exclusions,
  missing annotations, conflicts, and cleaning decisions.
- Never fabricate labels or silently force Track A to agree with Track B > 0.
- Preserve the existing cleaning policy unless a change is justified and tested.
- Prevent duplicate-text leakage across partitions. Fit learned preprocessing,
  including TF-IDF vocabulary and IDF weights, on training data only.
- Restrict neighbor retrieval to permitted training examples. Use out-of-fold
  retrieval features during training; never retrieve an example's own labels.
- Tune using validation data, not held-out test data. Compare models on the same
  evaluation examples and document partition policies.
- Main metrics are macro-F1 for Track A and Pearson correlation for Track B.
  Document per-emotion aggregation and metric edge-case handling.
- Record seeds, parameters, dataset revisions and fingerprints, source revision,
  and relevant software versions. Do not invent results or claim unrun checks.

## Reference baselines

- The Track A default is `FacebookAI/xlm-roberta-large`, with a pinned Hub
  revision. Checkpoint name and revision are configurable; switching to base or
  another RoBERTa-family checkpoint must not require rewriting the model.
- Match official-format in-memory frames: id, text, and named emotion columns.
  Preserve prediction IDs and row order. Missing emotion annotations are not
  negative labels; select the correct per-language schema explicitly.
- Use PyTorch and Transformers. Keep training reproducible, validate on development
  data, and select the best validation macro-F1 checkpoint, never a test checkpoint.
- Save self-contained model weights, tokenizer, label mapping, threshold, and
  metadata under `output/runs/<run-id>/checkpoint/`. Keep metrics and run metadata
  alongside it. Never overwrite an existing run. Intermediate Trainer checkpoints
  under the run's `training/` directory also include resumable training state.
- Memory-conscious defaults use 256 tokens, microbatch 2, accumulation 4,
  gradient checkpointing, and automatic CUDA BF16/FP16 where supported. These
  settings are not a guarantee of fitting every 16 GB GPU; verify on real hardware.
- Ground each track's baseline in the original task paper and relevant official
  implementation. Cite the source and record upstream revisions when used.
- Document architecture, training objective, selected languages, official splits,
  preprocessing, hyperparameters, metrics, commands, and deviations.
- Respect language-specific label sets; do not assume the English five-emotion
  schema applies everywhere. Follow Track C's cross-lingual training and
  evaluation constraints rather than treating it as ordinary in-language work.
- Use track-appropriate official inputs. Do not substitute the main project's
  jointly aligned English A+B data unless the referenced procedure requires it.
- Keep reference results distinct from main experiments. Compare scores directly
  only when tasks, languages, evaluation examples, and metrics are compatible.
- Describe results as exact reproductions only when experimental conditions
  justify that claim. Label modified procedures and document their differences.

## Testing and verification

Run relevant checks from the repository root. Standard main-project commands:

```bash
uv sync --locked
uv run --locked python -m pytest
uv run --locked python -m ruff check src tests
git diff --check
```

- Add tests for changed behavior, particularly annotation alignment, duplicate
  handling, leakage prevention, retrieval, and metric edge cases.
- Default tests must use synthetic data without network access or downloaded
  datasets. Keep small synthetic fixtures under `tests/fixtures/`.
- Report which checks ran and disclose failures or anything not verified.

## Generated files and Git hygiene

- Generated datasets belong in `data/`; analyses, trained models, predictions,
  and run artifacts belong in `output/`. Use distinct experiment run directories.
- Preserve `.gitkeep` exceptions in ignored directories. Do not commit virtual
  environments, caches, real dataset fixtures, or large model artifacts.
- Keep selected result summaries tracked under `reports/results/`.
- Preserve unrelated changes. Do not stage, commit, or discard work unless asked.

## Documentation

- Keep the README brief and operational, without university-project framing or
  descriptions of future model choices beyond the agreed project scope.
- Document only available commands as runnable. Keep the README's dataset
  preparation section as TODO until its documentation is requested.
- Reports use LaTeX, including the required AI-usage appendix. Do not introduce
  a report build toolchain without a separate request.
- Update relevant documentation when supported commands or setup requirements
  change. Do not imply planned functionality has already been implemented.
