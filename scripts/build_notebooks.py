"""Build notebook sources from the two walkthroughs; execute with check_notebooks.py."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown_cell(text: str, source: Path, number: int) -> dict:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", text)

    def link(match):
        label, target = match.groups()
        if target.startswith(("http:", "https:", "#")):
            return match.group(0)
        path, _, fragment = target.partition("#")
        resolved = (source.parent / path).resolve().relative_to(ROOT)
        suffix = f"#{fragment}" if fragment else ""
        return f"[{label}](https://github.com/datakim/lendrisk/blob/main/{resolved}{suffix})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)
    return {
        "cell_type": "markdown",
        "id": f"text-{number}",
        "metadata": {},
        "source": text.strip() + "\n",
    }


def code_cell(code: str, number: int, *, install: bool = False) -> dict:
    return {
        "cell_type": "code",
        "id": f"code-{number}",
        "metadata": {"tags": ["skip-execution"]} if install else {},
        "source": code.strip() + "\n",
        "execution_count": None,
        "outputs": [],
    }


def build(source: str, output: str) -> None:
    path = ROOT / source
    body = path.read_text()
    cells = [
        markdown_cell(
            "Run the installation cell once, then choose **Run all**. All data is synthetic. "
            "The saved outputs were executed against the local package.\n\n"
            "[Documentation](https://datakim.github.io/lendrisk/)",
            path,
            0,
        ),
        code_cell(
            '%pip install -q "git+https://github.com/datakim/lendrisk.git@v0.1.0a2" matplotlib',
            0,
            install=True,
        ),
    ]
    previous = 0
    for match in re.finditer(r"```(python|bash)\n(.*?)```", body, re.S):
        text = body[previous : match.start()]
        if text.strip():
            cells.append(markdown_cell(text, path, len(cells)))
        if match.group(1) == "python":
            cells.append(code_cell(match.group(2), len(cells)))
            if output.startswith("02") and "bins = OptimalBinning" in match.group(2):
                cells.append(
                    code_cell(
                        """import matplotlib.pyplot as plt

regular = bins.table().query("bin != 'Missing' and bin != 'Special'")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = [f"Bin {i+1}" for i in range(len(regular))]
axes[0].bar(labels, regular["event_rate"], color="#147d78")
axes[0].set_title("Observed default rate by bin")
axes[0].set_ylabel("Default share")
axes[1].bar(labels, regular["woe"], color="#147d78")
axes[1].axhline(0, color="black", linewidth=0.7)
axes[1].set_title("Weight of evidence by bin")
fig.tight_layout()
plt.show()""",
                        len(cells),
                    )
                )
            if output.startswith("02") and "points = model.score_points" in match.group(2):
                cells.append(
                    code_cell(
                        """fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(points, pd_hat, color="#147d78", alpha=0.6)
ax.set_xlabel("Score points")
ax.set_ylabel("Model PD")
ax.set_title("Synthetic test cohort: higher points mean lower model PD")
fig.tight_layout()
plt.show()""",
                        len(cells),
                    )
                )
        previous = match.end()
    if body[previous:].strip():
        cells.append(markdown_cell(body[previous:], path, len(cells)))
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    destination = ROOT / "notebooks" / f"{output}.ipynb"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(destination.relative_to(ROOT))


def main() -> None:
    build("docs/tutorials/revenue-financing.md", "01_revenue_financing")
    build("docs/tutorials/scorecards.md", "02_native_scorecard")


if __name__ == "__main__":
    main()
