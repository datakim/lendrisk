"""Execute notebook code with the current interpreter; skip only the install cell."""

import argparse
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Save freshly executed outputs")
    args = parser.parse_args()
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        with tempfile.TemporaryDirectory(prefix="lendrisk-notebook-") as directory:
            NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": directory}},
            ).execute()
        if args.write:
            nbformat.write(notebook, path)
        figures = sum(
            "image/png" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        if not figures:
            raise RuntimeError(f"No chart output produced by {path.name}")
        print(f"{path.name}: all tutorial cells executed; {figures} chart outputs")


if __name__ == "__main__":
    main()
