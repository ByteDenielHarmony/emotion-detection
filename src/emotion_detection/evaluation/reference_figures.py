"""Read-only figures for a saved reference-training suite.

Plotting uses saved epoch history, metadata, and development/test reports.
Nothing here trains, loads a model, downloads files, or scores new data.
"""

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

TRACKS = ("a", "b", "c")
FIGURE_NAMES = (
    "development_progress.png",
    "test_per_emotion.png",
    "performance_comparison.png",
)

# Organizers' English test baselines from the task overview paper
# (arXiv:2503.07269, Tables 2, 3, and 4). Development scores are not published.
ENGLISH_BASELINE_TEST = {"a": 0.708, "b": 0.641, "c": 0.375}

_TRACK_SPEC = {
    "a": {
        "metric": "macro_f1",
        "metric_label": "Macro-F1",
        "panel": "A development macro-F1",
        "per_emotion": "F1",
        "per_emotion_key": "f1",
        "development_split": "dev",
    },
    "b": {
        "metric": "pearson",
        "metric_label": "Mean Pearson",
        "panel": "B development mean Pearson",
        "per_emotion": "Pearson",
        "per_emotion_key": "pearson",
        "development_split": "dev",
    },
    "c": {
        "metric": "macro_f1",
        "metric_label": "Macro-F1",
        "panel": "C source-development macro-F1",
        "per_emotion": "F1",
        "per_emotion_key": "f1",
        "development_split": "source_dev",
    },
}


@dataclass(frozen=True)
class EpochPoint:
    epoch: float
    value: float


@dataclass(frozen=True)
class TrackSeries:
    track: str
    language: str
    metric_key: str
    metric_label: str
    panel_title: str
    per_emotion_label: str
    development_split: str
    points: tuple[EpochPoint, ...]
    selected_epoch: float
    selected_score: float
    stopped_early: bool
    test_labels: tuple[str, ...]
    test_scores: tuple[float, ...]
    test_aggregate: float


@dataclass(frozen=True)
class ReferenceSuite:
    run_id: str
    language: str
    tracks: tuple[TrackSeries, ...]


@dataclass(frozen=True)
class ComparisonRow:
    track: str
    metric_label: str
    development_score: float
    source_development: bool
    test_score: float
    baseline: float | None
    difference: float | None


@dataclass(frozen=True)
class ComparisonTable:
    rows: tuple[ComparisonRow, ...]
    baselines_applied: bool
    notes: tuple[str, ...]


def _close(left, right):
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-6)


def _fail(message):
    raise ValueError(message)


def _missing(path):
    return (
        f"Missing required artifact: {path}. This command only reads an existing "
        "run and does not train, download, or evaluate."
    )


def _read_json(path):
    if not path.is_file():
        raise ValueError(_missing(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not read {path}: {error}") from error


def _read_jsonl(path):
    if not path.is_file():
        raise ValueError(_missing(path))
    rows = []
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Could not read {path} line {number}: {error}") from error
    if not rows:
        raise ValueError(f"{path} has no epoch records")
    return rows


def _finite(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not a number") from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{description} is not a finite number")
    return number


def _language(metadata, track):
    try:
        language = metadata["data"]["provenance"]["language"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Track {track.upper()} metadata is missing data.provenance.language. "
            "Language is required so published English baselines are not assumed."
        ) from error
    if not isinstance(language, str) or not language:
        raise ValueError(
            f"Track {track.upper()} metadata has an empty language, so English "
            "baselines were not applied."
        )
    return language


def _history(records, track, spec):
    points = []
    seen = set()
    stops = []
    expected = spec["development_split"]
    for record in records:
        split = record.get("split")
        if split == "test":
            raise ValueError(
                f"Track {track.upper()} epoch_history.jsonl contains test "
                "performance. Development figures must not include test curves."
            )
        if split != expected:
            raise ValueError(
                f"Track {track.upper()} epoch history split is {split!r}, expected "
                f"{expected!r}. Source-development and target-development results "
                "are not interchangeable."
            )
        epoch = _finite(record.get("epoch"), f"Track {track.upper()} epoch")
        if any(math.isclose(epoch, previous, abs_tol=1e-9) for previous in seen):
            raise ValueError(f"Track {track.upper()} repeats epoch {epoch:g}")
        seen.add(epoch)
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or spec["metric"] not in metrics:
            raise ValueError(
                f"Track {track.upper()} epoch {epoch:g} has no {spec['metric']}. "
                "Loss curves are not a substitute."
            )
        points.append(
            EpochPoint(
                epoch,
                _finite(
                    metrics[spec["metric"]],
                    f"Track {track.upper()} epoch {epoch:g} {spec['metric']}",
                ),
            )
        )
        if "stop_requested" not in record:
            raise ValueError(
                f"Track {track.upper()} epoch {epoch:g} has no stop_requested flag, "
                "so early stopping cannot be marked from the saved history."
            )
        stops.append(bool(record["stop_requested"]))
    order = sorted(range(len(points)), key=lambda index: points[index].epoch)
    ordered_points = tuple(points[index] for index in order)
    ordered_stops = [stops[index] for index in order]
    if any(ordered_stops[:-1]):
        raise ValueError(
            f"Track {track.upper()} requested stopping before the final epoch."
        )
    return ordered_points, ordered_stops[-1]


def _selected(points, metadata, validation, summary_entry, track, spec):
    training = metadata.get("training")
    if not isinstance(training, dict):
        _fail(f"Track {track.upper()} metadata has no training record")
    if "best_epoch" not in training:
        raise ValueError(f"Track {track.upper()} metadata has no training.best_epoch")
    best_epoch = _finite(
        training["best_epoch"], f"Track {track.upper()} training.best_epoch"
    )
    matches = [
        point for point in points if math.isclose(point.epoch, best_epoch, abs_tol=1e-9)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Track {track.upper()} selected epoch {best_epoch:g} is not in "
            "epoch_history.jsonl."
        )
    selected = matches[0]
    if "best_metric" not in training or not _close(training["best_metric"], selected.value):
        raise ValueError(
            f"Track {track.upper()} metadata best_metric does not match epoch "
            f"{best_epoch:g} {spec['metric']} ({selected.value})."
        )
    if validation.get("split") != spec["development_split"]:
        raise ValueError(
            f"Track {track.upper()} validation_report.json split is "
            f"{validation.get('split')!r}, expected {spec['development_split']!r}."
        )
    if "epoch" not in validation or not math.isclose(
        _finite(validation["epoch"], f"Track {track.upper()} validation epoch"),
        best_epoch,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"Track {track.upper()} validation_report.json is not the selected "
            f"epoch {best_epoch:g}."
        )
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict) or spec["metric"] not in metrics:
        raise ValueError(
            f"Track {track.upper()} validation_report.json has no {spec['metric']}."
        )
    if not _close(metrics[spec["metric"]], selected.value):
        raise ValueError(
            f"Track {track.upper()} validation_report.json {spec['metric']} does not "
            "match the selected epoch in epoch_history.jsonl."
        )
    summary_score = summary_entry.get("development_score")
    if summary_score is not None and not _close(summary_score, selected.value):
        raise ValueError(
            f"summary.json development_score for track {track.upper()} does not "
            "match validation_report.json."
        )
    return selected


def _stopping(metadata, history_stopped, track):
    training = metadata["training"]
    if "stopped_early" not in training:
        raise ValueError(f"Track {track.upper()} metadata has no training.stopped_early")
    stopped_early = training["stopped_early"]
    if not isinstance(stopped_early, bool):
        _fail(f"Track {track.upper()} training.stopped_early is not true or false")
    # The trainer sets stop_requested when the run ends, including at the epoch
    # limit. Early stopping is the metadata flag, which must agree with that end.
    if stopped_early and not history_stopped:
        _fail(
            f"Track {track.upper()} metadata says training stopped early, but the "
            "final epoch did not request a stop."
        )
    reason = training.get("stopping_reason")
    if stopped_early and reason not in (None, "early_stopping"):
        raise ValueError(
            f"Track {track.upper()} stopped early but stopping_reason is {reason!r}."
        )
    if not stopped_early and reason == "early_stopping":
        raise ValueError(
            f"Track {track.upper()} stopping_reason is early_stopping but "
            "stopped_early is false."
        )
    return stopped_early


def _test_scores(report, summary_entry, track, spec):
    if report.get("split") not in (None, "test"):
        raise ValueError(
            f"Track {track.upper()} test_report.json split is {report.get('split')!r}, "
            "expected 'test'."
        )
    if report.get("track") not in (None, track):
        raise ValueError(
            f"Track {track.upper()} test_report.json says track {report.get('track')!r}."
        )
    labels = report.get("label_names")
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(name, str) and name for name in labels)
    ):
        raise ValueError(
            f"Track {track.upper()} test_report.json has no emotion names in label_names."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Track {track.upper()} test_report.json repeats an emotion name")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        _fail(f"Track {track.upper()} test_report.json has no metrics")
    prefix = spec["per_emotion_key"]
    scores = []
    for name in labels:
        key = f"{prefix}_{name}"
        if key not in metrics:
            raise ValueError(
                f"Track {track.upper()} test report has no {key}. Emotion names are "
                "taken from the artifact and are not filled in from an English list."
            )
        scores.append(_finite(metrics[key], f"Track {track.upper()} {key}"))
    if spec["metric"] not in metrics:
        raise ValueError(
            f"Track {track.upper()} test_report.json has no {spec['metric']}."
        )
    aggregate = _finite(
        metrics[spec["metric"]], f"Track {track.upper()} test {spec['metric']}"
    )
    summary_metrics = summary_entry.get("test_metrics") or {}
    if spec["metric"] in summary_metrics and not _close(summary_metrics[spec["metric"]], aggregate):
        raise ValueError(
            f"summary.json test {spec['metric']} for track {track.upper()} does not "
            "match test_report.json."
        )
    return tuple(labels), tuple(scores), aggregate


def _load_track(run_dir, track, summary):
    spec = _TRACK_SPEC[track]
    tracks = summary.get("tracks")
    entry = tracks.get(track) if isinstance(tracks, dict) else None
    if not isinstance(entry, dict):
        _fail(
            f"summary.json has no track {track.upper()} entry. A reference report "
            "needs completed tracks A, B, and C."
        )
    status = entry.get("status")
    if status != "completed":
        recorded = entry.get("error")
        detail = f" Recorded error: {recorded}." if recorded else ""
        raise ValueError(
            f"Track {track.upper()} is {status or 'missing'}, not completed.{detail} "
            "Figures are not drawn from an unfinished track."
        )
    child = run_dir / track
    test_path = child / "test_report.json"
    if entry.get("test_skipped") and not test_path.is_file():
        raise ValueError(
            f"Track {track.upper()} skipped final test evaluation, so "
            "test_per_emotion.png and performance_comparison.png cannot be drawn. "
            f"Missing {test_path}."
        )
    points, history_stopped = _history(
        _read_jsonl(child / "epoch_history.jsonl"), track, spec
    )
    metadata = _read_json(child / "metadata.json")
    if metadata.get("track") not in (None, track):
        raise ValueError(
            f"{child / 'metadata.json'} says track {metadata.get('track')!r}, "
            f"expected {track!r}."
        )
    validation = _read_json(child / "validation_report.json")
    selected = _selected(points, metadata, validation, entry, track, spec)
    labels, scores, aggregate = _test_scores(_read_json(test_path), entry, track, spec)
    return TrackSeries(
        track=track,
        language=_language(metadata, track),
        metric_key=spec["metric"],
        metric_label=spec["metric_label"],
        panel_title=spec["panel"],
        per_emotion_label=spec["per_emotion"],
        development_split=spec["development_split"],
        points=points,
        selected_epoch=selected.epoch,
        selected_score=selected.value,
        stopped_early=_stopping(metadata, history_stopped, track),
        test_labels=labels,
        test_scores=scores,
        test_aggregate=aggregate,
    )


def load_reference_suite(run_dir):
    """Load A/B/C development and test scores from a saved suite directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    summary_path = run_dir / "summary.json"
    if (run_dir / "epoch_history.jsonl").is_file() and not summary_path.is_file():
        raise ValueError(
            f"{run_dir} looks like a single-track run. report reference expects a "
            "suite directory with summary.json and child directories a/, b/, and c/."
        )
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        _fail("summary.json is not an object")
    tracks = tuple(_load_track(run_dir, track, summary) for track in TRACKS)
    languages = {track.language for track in tracks}
    if len(languages) != 1:
        detail = ", ".join(f"{track.track}={track.language}" for track in tracks)
        raise ValueError(
            f"Suite tracks disagree on language ({detail}). Refusing to build one "
            "comparison, and English baselines were not applied."
        )
    language = tracks[0].language
    summary_language = summary.get("language")
    if summary_language is not None and summary_language != language:
        raise ValueError(
            f"summary.json language is {summary_language!r}, but track metadata "
            f"says {language!r}."
        )
    return ReferenceSuite(run_id=run_dir.name, language=language, tracks=tracks)


def comparison_table(suite):
    """Build the performance table without drawing it.

    Published English test baselines are applied only when every track's saved
    language is ``eng``. Other languages keep the baseline and difference blank.
    """
    applied = suite.language == "eng"
    rows = []
    for track in suite.tracks:
        baseline = ENGLISH_BASELINE_TEST[track.track] if applied else None
        difference = None if baseline is None else track.test_aggregate - baseline
        rows.append(
            ComparisonRow(
                track=track.track.upper(),
                metric_label=track.metric_label,
                development_score=track.selected_score,
                source_development=track.development_split == "source_dev",
                test_score=track.test_aggregate,
                baseline=baseline,
                difference=difference,
            )
        )
    notes = [
        "Paper development scores are unavailable.",
        (
            "The asterisk marks source-development macro-F1. That score is not "
            "directly comparable to the target-language test score."
        ),
        "This setup is a modified reference implementation, not an exact reproduction.",
    ]
    if applied:
        notes.append(
            "Test difference is our test score minus the organizers' baseline test score."
        )
        notes.append(
            "English organizers' baseline test scores are A 0.708, B 0.641, and "
            "C 0.375 (arXiv:2503.07269, Tables 2-4)."
        )
    else:
        notes.append(
            "Published English baseline test scores were not applied because this "
            f"suite language is {suite.language!r}, not 'eng'."
        )
    return ComparisonTable(
        rows=tuple(rows), baselines_applied=applied, notes=tuple(notes)
    )


def _pyplot():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    plt = _pyplot()
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def render_development_progress(suite, path):
    """Plot recorded development scores, with no smoothing and no test curves."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    fig.suptitle(suite.run_id, fontsize=11, color="#374151")
    for axis, track in zip(axes, suite.tracks, strict=True):
        epochs = [point.epoch for point in track.points]
        values = [point.value for point in track.points]
        axis.plot(
            epochs,
            values,
            color="#1f4e79",
            marker="o",
            linewidth=1.6,
            label="Epoch score",
        )
        selected = next(
            point
            for point in track.points
            if math.isclose(point.epoch, track.selected_epoch, abs_tol=1e-9)
        )
        axis.scatter(
            [selected.epoch],
            [selected.value],
            s=46,
            marker="D",
            color="#c2410c",
            zorder=4,
            label="Selected checkpoint",
        )
        axis.annotate(
            f"selected\n{selected.value:.3f}",
            xy=(selected.epoch, selected.value),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            color="#9a3412",
        )
        if track.stopped_early:
            last = track.points[-1]
            axis.axvline(
                last.epoch,
                color="#6b7280",
                linestyle=":",
                linewidth=1.2,
                label="Early stopping",
            )
            same_point = math.isclose(last.epoch, selected.epoch, abs_tol=1e-9)
            axis.annotate(
                "early stop",
                xy=(last.epoch, last.value),
                xytext=(0, -18 if same_point else 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#374151",
            )
        low = min(values)
        high = max(values)
        span = high - low
        if span == 0:
            span = max(abs(high), 1.0) * 0.05 or 0.05
        axis.set_ylim(low - span * 0.15, high + span * 0.28)
        axis.set_xticks(epochs)
        axis.set_xticklabels([f"{epoch:g}" for epoch in epochs])
        axis.set_title(track.panel_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(track.metric_label)
        axis.grid(axis="y", linestyle=":", color="#d1d5db")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def render_test_per_emotion(suite, path):
    """Draw final test scores per emotion using names stored in the test reports."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2))
    fig.suptitle(f"{suite.run_id}: final test", fontsize=11, color="#374151")
    for axis, track in zip(axes, suite.tracks, strict=True):
        labels = list(track.test_labels)
        scores = list(track.test_scores)
        positions = list(range(len(labels)))
        axis.barh(positions, scores, color="#d6c6a5", edgecolor="#8a7a55", height=0.72)
        axis.axvline(
            track.test_aggregate,
            color="#111827",
            linestyle="--",
            linewidth=1.1,
            label=track.metric_label,
        )
        low = min([0.0, track.test_aggregate, *scores])
        high = max([0.0, track.test_aggregate, *scores])
        span = high - low or 1.0
        axis.set_xlim(low - span * 0.02, high + span * 0.22)
        for position, score in zip(positions, scores, strict=True):
            if score >= 0:
                axis.text(
                    score + span * 0.02,
                    position,
                    f"{score:.3f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                )
            else:
                axis.text(
                    score - span * 0.02,
                    position,
                    f"{score:.3f}",
                    va="center",
                    ha="right",
                    fontsize=9,
                )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.invert_yaxis()
        axis.set_title(
            f"({track.track}) Track {track.track.upper()}\n"
            f"Per-emotion {track.per_emotion_label}"
        )
        axis.set_xlabel(f"{track.metric_label}: {track.test_aggregate:.3f}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _cell(value, signed=False):
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.4f}"
    return f"{value:.4f}"


def render_performance_comparison(suite, path):
    """Draw the track comparison table and the limits of that comparison."""
    plt = _pyplot()
    table = comparison_table(suite)
    fig = plt.figure(figsize=(12.6, 4.9))
    fig.suptitle(
        "Modified reference implementation, not an exact reproduction",
        fontsize=13,
        color="#1f2937",
        y=0.98,
    )
    axis = fig.add_axes((0.02, 0.36, 0.96, 0.56))
    axis.axis("off")
    headers = [
        "Track",
        "Metric",
        "Our selected\ndev score",
        "Our test\nscore",
        "Paper baseline\ntest",
        "Test\ndifference",
    ]
    body = []
    for row in table.rows:
        development = _cell(row.development_score)
        if row.source_development:
            development += "*"
        baseline = "n/a" if row.baseline is None else f"{row.baseline:.3f}"
        body.append(
            [
                row.track,
                row.metric_label,
                development,
                _cell(row.test_score),
                baseline,
                _cell(row.difference, signed=True),
            ]
        )
    drawn = axis.table(
        cellText=body, colLabels=headers, loc="center", cellLoc="center"
    )
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(10)
    drawn.scale(1, 2.35)
    for (row_index, column), cell in drawn.get_celld().items():
        cell.set_edgecolor("#d6d3d1")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#312e81")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.18)
            continue
        cell.set_facecolor("#ffffff" if row_index % 2 else "#f5f5f4")
        if column == 5 and table.rows[row_index - 1].difference is not None:
            color = "#166534" if table.rows[row_index - 1].difference >= 0 else "#991b1b"
            cell.set_text_props(color=color)
    fig.text(
        0.03,
        0.04,
        "\n".join(table.notes),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#1f2937",
    )
    _save(fig, path)


def write_reference_figures(run_dir, output_dir, *, overwrite=False):
    """Write the three suite figures. Existing files are kept unless overwrite."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    run_resolved = run_dir.resolve()
    out_resolved = output_dir.resolve()
    if out_resolved == run_resolved or run_resolved in out_resolved.parents:
        raise ValueError(
            f"Refusing to write figures inside the run directory: {out_resolved}"
        )
    suite = load_reference_suite(run_resolved)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in FIGURE_NAMES]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing figures "
            f"({listed}). Pass --overwrite to replace them."
        )
    with tempfile.TemporaryDirectory(prefix=".figures-", dir=output_dir) as temporary:
        temp = Path(temporary)
        render_development_progress(suite, temp / FIGURE_NAMES[0])
        render_test_per_emotion(suite, temp / FIGURE_NAMES[1])
        render_performance_comparison(suite, temp / FIGURE_NAMES[2])
        for name in FIGURE_NAMES:
            (temp / name).replace(output_dir / name)
    return tuple(targets)
