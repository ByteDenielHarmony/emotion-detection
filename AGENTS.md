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
- Dataset preparation and transformer reference baselines for A/B/C exist,
  sharing one lifecycle and binary/intensity task strategies with synthetic tests.
  Inspect current code and distinguish implemented, tested, and trained models;
  a passing tiny-model test is not evidence of a pretrained-model benchmark run.
- The argparse app supports `emotion` and `python -m emotion_detection`: `doctor`,
  `data languages`, `data prepare`, read-only reference `data check`,
  `train reference` (A/B/C sequentially by default; `--track` selects one), and
  read-only `report reference`. `report reference` only reads an existing suite's
  epoch history, metadata, and saved development/test reports. It must not train,
  load a model, download, or evaluate. Do not advertise evaluation/prediction
  CLI commands until implemented.
- Track A reference manifests cover 28 languages (84 pinned source files). Use
  `available_reference_languages(track=...)` for discovery, not duplicated CLI
  lists or an assumed common language set across tracks. C discovery lists target
  languages, not protocol readiness: validate its source-selection protocol too.
  The pinned release lacks Romanian B training and a default C source set for
  Sundanese; never substitute other-track data or skip those failures silently.
  English excludes disgust, Afrikaans excludes surprise, and the other registered
  languages use all six labels. Respect the declared schema, not just label count.

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
- Keep help/version lightweight: do not import PyTorch/Transformers or acquire
  data/models while constructing the parser. Imports belong in command handlers.
- CLI training orchestration lives in `experiments/reference.py`; dataset file
  I/O remains in `datasets/`, not in classifiers or argument-parsing code.
- Training's default `--track all` means A/B/C for one selected language, not all
  languages. Preflight every requested manifest, protocol, and prepared dataset
  before loading any model. Never silently skip unsupported track/language pairs.
- Each track starts fresh from the configured pretrained source; never pass A/B
  trained weights into C automatically. Reject known project warm starts containing
  target-language supervision, and document external-checkpoint provenance.
- Run one model at a time and release model/trainer references plus CUDA caches
  between tracks. Suites keep separate a/b/c children and a status/metric summary;
  fail fast while preserving completed results. Do not aggregate F1 and Pearson.
- `prepare_reference_dataset` acquires and persists data;
  `load_prepared_reference_dataset` only reads, validates provenance/fingerprints,
  and checks schemas and split separation. The historical `load_reference_dataset`
  still prepares data for compatibility. Never call it as a read-only loader.
- Training requires prepared data unless `--prepare-data` is explicitly passed.
  Device/argument validation must happen before that preparation. Do not silently
  repair stale data or fall back from explicitly requested CUDA. Never score test
  data during training or checkpoint selection.
- Use Python modules and CLI commands, not notebooks. Do not introduce an
  experiment configuration framework or extra infrastructure without a request.
- Reference baselines live under `src/emotion_detection/reference_baselines/`.
  Reuse checkpoint/tokenizer utilities and in-memory adapters across tracks,
  while keeping objectives and experiment orchestration track-specific.
- `base.py` owns the shared lifecycle; `tasks.py` owns validation, target/head
  encoding, loss, decoding, and diagnostics. Keep the three track wrappers thin
  and preserve the public Track A API and legacy checkpoint loading.
- Persist model emotion names, native evaluation emotion names, and head-output
  slots separately. B has four output slots per emotion; C may have six model
  emotions but only five native target emotions. Do not equate every model's
  `config.id2label` with its native evaluation labels. Use `load_reference_model`
  for track dispatch and retain compatibility with legacy A checkpoints.
- Model APIs receive already-loaded data. Do not add persistent data loading,
  downloading, or preprocessing to the model code. Keep these responsibilities
  in the dataset package.

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

- All reference tracks default to `FacebookAI/xlm-roberta-large`, with a pinned Hub
  revision. Checkpoint name and revision are configurable; switching to base or
  another RoBERTa-family checkpoint must not require rewriting the model.
- Match official-format in-memory frames: id, text, and named emotion columns.
  Preserve prediction IDs and row order. Missing emotion annotations are not
  negative labels; select the correct per-language schema explicitly.
- Use PyTorch and Transformers. Keep training reproducible and validate on
  development data only. `--monitor auto` resolves to macro-F1 for A/C and mean
  Pearson for B; explicit incompatible monitors must fail before preparation.
  `--monitor loss` explicitly selects minimum development loss instead. Never
  select checkpoints or stop training using test performance.
- B uses four logits per emotion and cross-entropy, decoding integer classes
  0-3 for official-style evaluation. Report Pearson, MAE, and 4x4 confusion
  matrices. Undefined correlations contribute zero to the fixed-label mean and
  are explicitly flagged, following BRIGHTER's convention.
- C fits only source train/source_dev, never target_dev or target training.
  Respect six model outputs versus target-native evaluation labels, composite
  `(source_language, id)` identities, and known_<emotion> masks. Unknown annotations
  contribute nothing to loss, weights, or metrics; any numeric tensor placeholder
  is masked and must not alter the nullable input DataFrame. Mark entirely
  unobserved labels undefined instead of inventing negative observations.
- Early stopping is opt-in via `--early-stopping-patience`; the default fixed
  two-epoch recipe remains available. Checkpoint selection and stopping share
  one monitor. Min-delta controls patience resets, not absolute-best selection.
- The CLI experiment runner evaluates the held-out test set once by default,
  only after `fit` has selected/restored the best development checkpoint and
  exported it. `--skip-test` disables this phase for development runs. Model
  `fit` receives training and selection frames only (`source_dev` for C); test
  results never alter selection or thresholds. `--skip-test` applies to every
  selected track in a suite.
- Keep final testing separate in `evaluation/reference.py`, receiving data in
  memory. Save non-overwriting `test_metrics.json`, `test_report.json`, and
  `test_predictions.csv` without modifying development reports or checkpoint
  weights/metadata. Match recorded CUDA BF16/FP16 evaluation precision where
  supported (CPU is FP32), and record actual precision and the test fingerprint.
- Do not retrospectively evaluate existing real runs without an explicit request.
  Tests must use synthetic held-out data. Seeing test results repeatedly does not
  authorize tuning on them; document the holdout limitation clearly.
- Save self-contained model weights, tokenizer, label mapping, threshold, and
  metadata in each track run's `checkpoint/`: `output/runs/<run-id>/checkpoint/`
  for single-track runs, or `output/runs/<suite-id>/<a|b|c>/checkpoint/` for suites.
  A suite `--run-id` names the group, not an individual child. Keep metrics and
  run metadata alongside it. Never overwrite an existing run. Trainer rotation keeps at most
  two intermediate checkpoints under that new run's `training/`, protecting best
  and latest and retaining optimizer state. Do not implement broad cleanup across
  runs or delete existing checkpoints outside the current training directory.
- Print task-appropriate development metrics and confusion matrices each epoch,
  reusing predictions from validation. Matrix rows are actual classes and columns
  are predicted classes: absent/present for A/C, or 0,1,2,3 for B. Keep all reports
  in `epoch_history.jsonl` regardless of checkpoint rotation, and put the selected report in
  `validation_report.json`. The final best-model evaluation is not another epoch.
  A/C matrices are 2x2; B matrices are 4x4. C reports and metadata must
  identify selection as source_dev rather than target development.
- Record the monitor, selected best metric/epoch, actual completed epochs, and
  stopping reason. Do not report loss as F1/Pearson: track-appropriate best metric
  and selected checkpoint scores are recorded separately.
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
- Pytest discovery is restricted to `tests/`; nested checkouts are not test suites
  of this project and must not be collected or committed accidentally.
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
- Document only available commands as runnable. Keep setup, data-preparation,
  and training examples aligned with the CLI and retain pip-only instructions.
- Reports use LaTeX, including the required AI-usage appendix. Do not introduce
  a report build toolchain without a separate request.
- Update relevant documentation when supported commands or setup requirements
  change. Do not imply planned functionality has already been implemented.
