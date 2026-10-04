# Learn lendrisk with notebooks

The notebooks contain executed outputs from synthetic examples. Each starts
with a tagged GitHub installation cell and runs independently.

| Notebook | What you learn | Open without local setup |
| --- | --- | --- |
| [Revenue financing](01_revenue_financing.ipynb) | History vs. future paths, terms, daily payments, sales shocks, cash, floors and milestones | [Open in Colab](https://colab.research.google.com/github/datakim/lendrisk/blob/main/notebooks/01_revenue_financing.ipynb) |
| [Native scorecard](02_native_scorecard.ipynb) | Numerical bins, WoE, train/test separation, score points, model diagnostics and contributions | [Open in Colab](https://colab.research.google.com/github/datakim/lendrisk/blob/main/notebooks/02_native_scorecard.ipynb) |

In Colab, run the installation cell first, then choose **Runtime → Run all**.
The model and simulation run in that notebook's Python environment.

For local use, clone the repository, install `jupyterlab` and `matplotlib`, then
run `jupyter lab` from the repository root. The first cell installs the tagged alpha.

Maintainers can regenerate notebook sources with `python scripts/build_notebooks.py`
and save fresh outputs with `python scripts/check_notebooks.py --write` after
installing the development/documentation dependencies. CI executes the code cells
against the current checkout and skips the installation cell.
