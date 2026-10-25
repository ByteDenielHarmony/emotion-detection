"""Readable classification reports without plotting or additional inference."""


def format_classification_report(record, *, heading):
    metrics = record["metrics"]
    position = ""
    if record.get("epoch") is not None:
        position = f" - epoch {record['epoch']:g}, step {record['global_step']}"
    monitor_label = (
        "Development selection monitor" if record.get("split") == "test" else "Monitor"
    )
    if "pearson" in metrics:
        lines = [
            f"\n{heading}{position}",
            f"Loss: {metrics['loss']:.4f} | Pearson: {metrics['pearson']:.4f} | MAE: {metrics['mae']:.4f}",
            f"{monitor_label}: {record['monitor']} ({record['direction']})",
            f"{'Emotion':<14} {'Pearson':>10} {'MAE':>10} {'Accuracy':>10} {'N':>9}",
        ]
        for name in record["label_names"]:
            lines.append(
                f"{name:<14} {metrics[f'pearson_{name}']:>10.4f} {metrics[f'mae_{name}']:>10.4f} {metrics[f'accuracy_{name}']:>10.4f} {record['support'][name]:>9}"
            )
        lines.append(
            "Intensity confusion matrices: rows=actual 0/1/2/3, columns=predicted 0/1/2/3"
        )
        for name, matrix in record["confusion_matrices"].items():
            lines.append(
                f"{name}:\n"
                + "\n".join(" ".join(f"{value:6}" for value in row) for row in matrix)
            )
        if record.get("undefined_correlations"):
            lines.append(
                "Undefined Pearson (reported as 0): "
                + ", ".join(record["undefined_correlations"])
            )
        return "\n".join(lines)

    def number(value):
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        f"\n{heading}{position}",
        (
            f"Loss: {metrics['loss']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Micro-F1: {metrics['micro_f1']:.4f}"
        ),
        f"{monitor_label}: {record['monitor']} ({record['direction']})",
        f"{'Emotion':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}",
    ]
    for name in record["label_names"]:
        lines.append(
            f"{name:<14} {number(metrics[f'precision_{name}']):>10} "
            f"{number(metrics[f'recall_{name}']):>10} {number(metrics[f'f1_{name}']):>10} "
            f"{record['support'][name]:>9}"
        )
    if record.get("undefined_emotions"):
        lines.append(
            "Unobserved labels (excluded from macro-F1): "
            + ", ".join(record["undefined_emotions"])
        )
    lines.append(
        "Confusion matrices: rows=actual, columns=predicted (absent / present)"
    )
    for name in record["label_names"]:
        (tn, fp), (fn, tp) = record["confusion_matrices"][name]
        lines.extend(
            [
                f"{name}:",
                f"{'':16} {'pred absent':>12} {'pred present':>13}",
                f"{'actual absent':16} {tn:>12} {fp:>13}",
                f"{'actual present':16} {fn:>12} {tp:>13}",
            ]
        )
    return "\n".join(lines)


def format_development_report(record, *, selected=False):
    if record.get("split") == "source_dev":
        heading = (
            "Selected checkpoint source development metrics"
            if selected
            else "Source development metrics"
        )
        return format_classification_report(record, heading=heading)
    heading = (
        "Selected checkpoint development metrics" if selected else "Development metrics"
    )
    return format_classification_report(record, heading=heading)
