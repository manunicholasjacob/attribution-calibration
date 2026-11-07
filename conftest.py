"""Present so that `python -m pytest tests -q` works in a clean clone before anything is installed.

With pytest's default import mode, the directory holding the topmost conftest is put on the import
path. A file here therefore makes `import attrcal` resolve to the source tree, which is what a
reader who has just cloned the repository will try first.

The alternative, inserting a path inside the test files, is what made the predecessor project
unreproducible on anyone else's machine, and `tests/test_packaging.py` asserts that no file in this
repository does it.
"""
