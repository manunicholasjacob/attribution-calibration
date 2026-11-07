# Contributing

This package was written to be reused, so contributions are welcome. It is small on purpose and the
constraints below are what keep it auditable.

## Getting set up

```bash
pip install -e ".[test]"
python -m pytest tests -q
python demo.py
```

Python 3.10 or newer. `numpy` is the only runtime dependency.

## What a change needs

**A test that asserts a behaviour, not a value.** Every existing test states something that should
be true of the harness: a detector that certifies everything scores as badly as possible on nulls
while having perfect recall; all five floor statuses are reachable; `permute` and `shift` disagree
on turn-ordered input and agree on a log without turn-taking. A test that pins a number without
saying why that number is right does not help anyone reading it later.

**Nothing new in `install_requires`.** numpy and the standard library. A harness whose argument is
that other people's error rates are unverified should be short enough to read, and each dependency
is one more thing a reader has to take on trust. `tests/test_packaging.py` enforces this by walking
each file's syntax tree, so an undeclared import fails the build rather than slipping through.

**No absolute paths, no `sys.path` manipulation, pure ASCII, no em dashes.** Also enforced by
`tests/test_packaging.py`. CI runs the suite on Linux and Windows across Python 3.10 to 3.13.

## Adding a surrogate

The point of the four existing nulls is that they differ only in which nuisance structure they
keep, which is what makes "a published error rate that does not name its null is not a rate" into
something you can check rather than something you have to believe. A new surrogate should say, in
its docstring and in the README table, exactly which structure it preserves and which it destroys,
and what question that lets a user answer.

If it is a null that is **invalid** in some regime, say so and keep it anyway, as `shift` is kept.
Being able to measure a bad null against a good one is more useful than being protected from it.

## Credit

If your change implements a published method, cite it in the code, in the README's Prior work
section, and in `CITATION.cff`. The package's own central idea is Runge's, adapted by Shorten,
Spinney and Lizier, and that is stated in all three places; anything added should meet the same bar.

## Sign-off

Commits carry a Developer Certificate of Origin sign-off:

```bash
git commit -s -m "your message"
```

That is `Signed-off-by: Your Name <your@email>`, which says you have the right to submit the work
under the project's MIT licence.
