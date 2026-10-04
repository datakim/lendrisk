# Upstream provenance and modifications

## Reviewed source

OptBinning 1.0.0, maintained by Guillermo Navas-Palencia:
[repository](https://github.com/guillermo-navas-palencia/optbinning),
[distribution](https://pypi.org/project/optbinning/1.0.0/).
The reviewed installed distribution contains Apache 2.0 licensing and copyright
attribution to the upstream author.

Code inspected: `binning/cp.py`, `binning/model_data.py`,
`binning/prebinning.py`, and `binning/transformations.py`.
Selected binary model-data calculations and CP-SAT solver setup were adapted
into `lendrisk/_binning_solver.py`. Original author attribution is retained in
its header and the package's NOTICE file.

## What changed

- Retain reversed cumulative event/nonevent counts and class-normalized IV
  candidate calculations from the upstream binary model-data construction.
- Replace triangular cumulative-assignment variables with explicit contiguous
  interval indicators and exact-cover constraints.
- Filter size and class-count violations before creating solver variables.
- Compare adjacent event rates through count cross-products to avoid integer
  quantization of the monotonicity constraint.
- Use an IV integer objective rounded at 1e8 scale; expose solver status and
  the absolute gap in that scaled objective.
- Raise when no feasible partition exists, and clear previously fitted state.
- Provide a focused numerical/binary public estimator, pandas tables, dedicated
  missing/special buckets, and a native multi-variable process and scorecard.

This is a maintained adaptation for a narrower initial use case. No speed or
accuracy improvement over upstream is claimed without comparative benchmarks.
The chosen boundaries optimize the specified prebin search space, not all
possible real-valued boundaries. Constraints and numerical conventions are
covered by exhaustive small-problem tests.

## License and attribution

The current distribution is Apache-2.0. `LICENSE` includes the upstream license
text and `NOTICE` identifies adapted files and authors. OptBinning is used only
as an optional development reference; runtime imports do not depend on it.

## Reviewed source file hashes

SHA-256 from the installed OptBinning 1.0.0 distribution:

- `binning/model_data.py`: `d911be131a7bc16a2c169d88bae8b0ec72c3044a79b26889bf6cff9b8b87edfa`
- `binning/cp.py`: `299b86415bbfe4921a53d1fb3452859bfe259f7f8f681522d3fd3507f0dd8e7d`
