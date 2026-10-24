"""Readable development reports without plotting or additional inference."""


def format_development_report(record, *, selected=False):
    metrics = record["metrics"]
    heading = (
        "Selected checkpoint development metrics" if selected else "Development metrics"
    )
    lines = [
        f"\n{heading} - epoch {record['epoch']:g}, step {record['global_step']}",
        (
            f"Loss: {metrics['loss']:.4f} | Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Micro-F1: {metrics['micro_f1']:.4f}"
        ),
        f"Monitor: {record['monitor']} ({record['direction']})",
        f"{'Emotion':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}",
    ]
    for name in record["label_names"]:
        lines.append(
            f"{name:<14} {metrics[f'precision_{name}']:>10.4f} "
            f"{metrics[f'recall_{name}']:>10.4f} {metrics[f'f1_{name}']:>10.4f} "
            f"{record['support'][name]:>9}"
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
