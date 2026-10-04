# Contributing

Start with an issue describing the analyst question, expected inputs, calculation
assumptions, and a small independently checkable example. Use synthetic or
publicly licensed data. Keep upstream comparison dependencies out of runtime imports.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs,reference]"
python -m pytest --cov=lendrisk --cov-report=term-missing
ruff check .
ruff format --check .
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Upstream comparison tests skip when OptBinning is absent. Native binning and
scorecards must pass with core/dev dependencies alone. Add regression tests for financial
behavior, edge cases, input contracts, or bugs; prefer reference calculations
and invariants to tests that simply repeat implementation steps.

## Alpha release

1. Run CI and verify the built wheel in a clean environment with examples.
2. Update package/version documentation and changelog together.
3. Commit and tag a version such as `v0.1.0a1`, then push the tag to GitHub.
4. GitHub installations can pin that tag immediately.

PyPI publication has not been configured. When ready, claim the package name,
configure PyPI Trusted Publishing for the repository and a protected release
environment, verify a TestPyPI upload, then publish the checked wheel/sdist.
Do not place credentials in source or workflow files. Version tags are immutable
release references; new fixes receive a new version.
