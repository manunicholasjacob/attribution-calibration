# Changelog

Notable changes to `attribution-calibration`. This project follows
[semantic versioning](https://semver.org/).

## Unreleased

### Added

- **Prior work section in the README, and a `references:` block in `CITATION.cff`**, crediting
  Runge (2018) for the local permutation scheme and Shorten, Spinney and Lizier (2021) for adapting
  it to continuous-time event data. The package's `permute` and `blockpermute` nulls are that idea
  applied to turn-ordered multi-agent logs. The first release described the method with no citation
  at all, which in a package pitched as reusable infrastructure reads as a claim of originality.
  The README now also separates code independence, which the packaging tests assert mechanically,
  from intellectual provenance, which they cannot.
- **Continuous integration** across Linux and Windows on Python 3.10 to 3.13. The README claimed "a
  clean clone runs on Linux unchanged"; it had only ever been run on Windows.
- **`CONTRIBUTING.md`**, covering the test command, the no-new-dependencies rule, what a new
  surrogate has to document, and the DCO sign-off.
- **Links to the two sibling repositories**, so the 0.90 figure that motivates the package is
  checkable rather than asserted.
- The `.zenodo.json` warning comment in `CITATION.cff`, matching the sibling repositories: a
  `.zenodo.json` silently overrides `CITATION.cff` when Zenodo archives a release.

### Fixed

- The README's install block said `pip install -e .` followed by `python -m pytest tests -q`, but
  `pytest` is an optional extra, so the second line failed on a clean machine. It is now
  `pip install -e ".[test]"`.

## 0.1.0 - 2026-11-06

First release. 192 tests, numpy and the standard library only.

- `calibrate()`: family-wise error measured on surrogates built from the user's own log, a power
  curve from a spike-in positive control, and a detection floor reported with an explicit status
  rather than a bare number. Every rate carries a Wilson interval and its denominator.
- Four surrogates, differing only in which nuisance structure they preserve: `permute`,
  `blockpermute`, `rotate`, and `shift`. `shift` is the conventional circular-shift recipe,
  implemented faithfully rather than charitably because it is **invalid on turn-ordered logs** and
  the point is to measure that rather than assert it. `null_index()` deliberately refuses to express
  `shift`.
- `audit_null()`: demonstrates on the user's own log why `shift` fails, instead of asking them to
  take it on faith.
- `synth_log()`: a generator with a knob for turn-taking and one for payload drift, the two
  properties the surrogate comparison depends on.
