"""Tests about the repository itself, so that a clean clone on another machine behaves.

Three of the project's own hard rules are checkable mechanically and are therefore checked here
rather than trusted: no absolute drive-letter paths anywhere, because the package has to run on
Linux from a clean clone; no dash variants or invisible Unicode, because the author writes ASCII
and a stray one is an edit nobody made on purpose; and no dependency on the private neighbouring
repositories, because this package is meant to stand alone.

Every banned literal in this file is assembled from pieces at run time. Written out, each one would
appear in the file being scanned and the check would fail on its own source, which is the classic
way a rule like this gets quietly disabled.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SELF = pathlib.Path(__file__).resolve()
TEXT_SUFFIXES = {".py", ".md", ".toml", ".cff", ".txt", ".cfg", ".yml", ".yaml"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", ".ruff_cache",
             ".eggs"}

DASHES = {"\u2014": "em dash", "\u2013": "en dash", "\u2212": "minus sign",
          "\u2012": "figure dash", "\u2015": "horizontal bar"}

#: Zero-width and direction-marking characters. They survive copy and paste, are invisible in a
#: diff, and have no business in source or documentation.
INVISIBLES = {"\u200b": "zero-width space", "\u200c": "zero-width non-joiner",
              "\u200d": "zero-width joiner", "\u2060": "word joiner",
              "\ufeff": "byte order mark", "\u00a0": "non-breaking space",
              "\u200e": "left-to-right mark", "\u200f": "right-to-left mark"}

#: A drive-letter path needs a single letter, a separator, a directory name and another separator.
#: The lookbehind keeps a URL scheme such as the one in "https://example.com" from matching, and
#: requiring a second separator keeps an escape sequence inside a string literal from matching.
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[A-Za-z0-9_.$-]{2,}[\\/]")

#: Modules that may be imported. Everything else is either a third-party dependency this package
#: does not declare or one of the private neighbouring repositories it must not touch.
ALLOWED_IMPORTS = {
    "attrcal", "numpy", "pytest",
    "__future__", "abc", "argparse", "ast", "collections", "dataclasses", "functools",
    "importlib", "inspect", "itertools", "json", "math", "os", "pathlib", "random", "re",
    "sys", "textwrap", "time", "typing", "warnings",
}

#: The private projects the predecessor benchmark imported from, which made its evaluator
#: unreproducible from a clean clone. Named here so the independence is asserted, not promised.
PRIVATE_NEIGHBOURS = ("swarm_te", "swarm_load", "fastte", "bench_gen", "bench_eval",
                      "swarm_prep", "swarm_sprint", "text_pool")


def repo_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info")
               for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            yield path


def python_files():
    return [p for p in repo_files() if p.suffix == ".py"]


def read(path):
    return path.read_text(encoding="utf-8")


def imported_names(path):
    """Top-level module names imported by a file, read from its syntax tree.

    A regular expression over the source matched prose: a docstring sentence beginning with the
    word "from" was reported as an undeclared dependency. The syntax tree cannot make that mistake.
    """
    out = set()
    for node in ast.walk(ast.parse(read(path))):
        if isinstance(node, ast.Import):
            out.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_the_repository_has_files_to_check():
    assert len(list(repo_files())) >= 10
    assert len(python_files()) >= 10


def test_no_absolute_drive_letter_paths_anywhere():
    """A hardcoded path is how the predecessor project ended up unreproducible from a clean clone."""
    offenders = []
    for path in repo_files():
        for i, line in enumerate(read(path).splitlines(), 1):
            if path == SELF and "ABSOLUTE_PATH" in line:
                continue
            if ABSOLUTE_PATH.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:80]}")
    assert not offenders, "absolute paths found:\n" + "\n".join(offenders)


def test_the_absolute_path_check_actually_matches_an_absolute_path():
    """A rule that matches nothing passes trivially, so the matcher is checked against examples.

    Each example is assembled from pieces, because written out it would appear in this file and the
    check above would then fail on its own test.
    """
    drive, colon, back, fwd = "C", ":", chr(92), "/"
    for bad in (drive + colon + back + "dev" + back + "thing",
                "D" + colon + fwd + "data" + fwd + "logs" + fwd + "x.json",
                'open("' + drive + colon + fwd + 'users' + fwd + 'me' + fwd + 'log.csv")'):
        assert ABSOLUTE_PATH.search(bad), bad
    for good in ("https" + colon + "//example.com/x",
                 'print("found' + colon + back + 'n")',
                 "ratio 3" + colon + "1 overall",
                 "." + fwd + "relative" + fwd + "path",
                 "attrcal" + fwd + "nulls.py"):
        assert not ABSOLUTE_PATH.search(good), good


def test_no_import_path_manipulation_anywhere():
    """A path insert is how an import of a private neighbouring repository gets smuggled in and
    then fails on somebody else's machine."""
    pattern = re.compile(r"sys" + re.escape(".") + r"path" + re.escape(".")
                         + r"(insert|append|extend)")
    offenders = [str(p.relative_to(ROOT)) for p in python_files() if pattern.search(read(p))]
    assert not offenders, f"import path manipulation in {offenders}"


def test_nothing_imports_the_private_neighbouring_projects():
    offenders = []
    for path in python_files():
        if path == SELF:
            continue
        for name in sorted(set(PRIVATE_NEIGHBOURS) & imported_names(path)):
            offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "\n".join(offenders)


def test_only_numpy_and_the_standard_library_are_imported():
    offenders = []
    for path in python_files():
        for name in sorted(imported_names(path) - ALLOWED_IMPORTS):
            offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "undeclared dependencies:\n" + "\n".join(offenders)


@pytest.mark.parametrize("char,label", sorted(DASHES.items()))
def test_no_dash_variants_anywhere(char, label):
    offenders = []
    for path in repo_files():
        if path == SELF:
            continue
        body = read(path)
        if char in body:
            line = next(i for i, ln in enumerate(body.splitlines(), 1) if char in ln)
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, f"{label} found in {offenders}"


@pytest.mark.parametrize("char,label", sorted(INVISIBLES.items()))
def test_no_invisible_unicode_anywhere(char, label):
    offenders = [str(p.relative_to(ROOT)) for p in repo_files()
                 if p != SELF and char in read(p)]
    assert not offenders, f"{label} found in {offenders}"


def test_every_file_including_this_one_is_pure_ascii():
    """The strongest form of the rule, and it covers this file too: an escape such as the one in
    the tables above is itself ASCII, so nothing here is exempt."""
    offenders = []
    for path in repo_files():
        try:
            read(path).encode("ascii")
        except UnicodeEncodeError as exc:
            offenders.append(f"{path.relative_to(ROOT)}: {exc.reason} at offset {exc.start}")
    assert not offenders, "non-ascii content:\n" + "\n".join(offenders)


def test_no_ai_attribution_or_coauthor_trailers_anywhere():
    banned = ("co" + "-authored-by", "generated with " + "claude", "generated by " + "claude",
              "robot face emoji placeholder".replace(" ", "-"))
    offenders = []
    for path in repo_files():
        if path == SELF:
            continue
        body = read(path).lower()
        for phrase in banned:
            if phrase in body:
                offenders.append(f"{path.relative_to(ROOT)} contains {phrase!r}")
    assert not offenders, "\n".join(offenders)


def test_the_required_metadata_files_are_present():
    for name in ("pyproject.toml", "README.md", "LICENSE", "requirements.txt", "CITATION.cff",
                 "demo.py"):
        assert (ROOT / name).is_file(), f"{name} is missing"
    assert (ROOT / "attrcal" / "__init__.py").is_file()


def test_the_citation_file_is_valid_cff_1_2_0_with_the_fields_that_are_commonly_omitted():
    """`date-released` rather than `year`, and `version`, `repository-code` and `url` present. A CFF
    file that fails validation is worse than none, because Zenodo and GitHub both parse it and a
    silent failure means the software is cited by whatever they guess instead."""
    body = read(ROOT / "CITATION.cff")
    assert "cff-version: 1.2.0" in body
    assert re.search(r'^date-released:\s*"?\d{4}-\d{2}-\d{2}"?\s*$', body, re.MULTILINE)
    assert not re.search(r"^year:", body, re.MULTILINE), "CFF 1.2.0 uses date-released, not year"
    for field in ("version:", "repository-code:", "url:", "license: MIT", "type: software",
                  "authors:", "message:", "abstract:"):
        assert field in body, f"CITATION.cff is missing {field}"
    assert "manunicholasjacob/attribution-calibration" in body
    assert "family-names: Jacob" in body
    assert "given-names: Manu Nicholas" in body
    assert "manunicholasjacob@gmail.com" in body
    assert "affiliation: Independent Researcher" in body


def test_the_licence_is_mit_and_names_the_author():
    body = read(ROOT / "LICENSE")
    assert "MIT License" in body
    assert "Manu Nicholas Jacob" in body
    assert "WITHOUT WARRANTY OF ANY KIND" in body.upper()


def test_the_declared_version_matches_the_package_in_both_metadata_files():
    import attrcal
    assert f'version = "{attrcal.__version__}"' in read(ROOT / "pyproject.toml")
    assert f"version: {attrcal.__version__}" in read(ROOT / "CITATION.cff")


def test_the_requirements_file_declares_numpy_and_pulls_in_nothing_heavier():
    """The comment may discuss scikit-learn; an uncommented requirement line may not name it."""
    lines = [ln.strip() for ln in read(ROOT / "requirements.txt").splitlines()]
    active = [ln for ln in lines if ln and not ln.startswith("#")]
    assert any(ln.startswith("numpy") for ln in active)
    assert not any("scikit" in ln or "sklearn" in ln or "scipy" in ln for ln in active)


def test_the_package_declares_the_same_dependency_as_the_requirements_file():
    body = read(ROOT / "pyproject.toml")
    assert "numpy>=" in body
    assert "scikit-learn" not in body


def test_the_public_api_is_importable_and_exports_what_the_readme_promises():
    import attrcal
    for name in ("calibrate", "Report", "PlantSpec", "compare_nulls", "audit_null", "Rate",
                 "detection_floor", "wilson", "make_null", "plant_copycat",
                 "adjusted_rand_index", "symbolisation_check", "synth_log", "detectors"):
        assert hasattr(attrcal, name), f"attrcal.{name} is missing"
        assert name in attrcal.__all__, f"attrcal.{name} is not in __all__"


def test_everything_in_all_actually_exists():
    import attrcal
    assert all(hasattr(attrcal, name) for name in attrcal.__all__)


def test_every_public_module_and_export_has_a_real_docstring():
    """Docstrings here are meant to explain why a choice was made, so a one-line restatement of the
    function name would defeat the point; a length floor is the cheapest proxy for that."""
    import importlib
    import inspect
    import attrcal
    for mod_name in ("calibrate", "events", "nulls", "plant", "report", "stats", "symbols",
                     "synth", "detectors"):
        mod = importlib.import_module(f"attrcal.{mod_name}")
        assert mod.__doc__ and len(mod.__doc__) > 200, f"{mod_name} needs a real module docstring"
    for name in attrcal.__all__:
        obj = getattr(attrcal, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            assert obj.__doc__ and len(obj.__doc__) > 40, f"attrcal.{name} needs a real docstring"


def test_the_readme_documents_the_four_nulls_and_the_floor_statuses():
    body = read(ROOT / "README.md")
    import attrcal
    for kind in attrcal.NULLS:
        assert kind in body, f"the README does not mention the {kind} null"
    for status in ("bracketed", "below_grid", "never_reached", "no_data", "non_monotone"):
        assert status in body, f"the README does not mention the {status} floor status"
    assert "Manu Nicholas Jacob" in body
    assert "MIT" in body
