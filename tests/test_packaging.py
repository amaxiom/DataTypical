"""
Release metadata consistency.

Since v0.8.2 the version is written down once, as ``datatypical.__version__``.
``to_config()`` stamps fits with it and the packaging metadata reads it. The
remaining places it appears are prose (the module docstrings, the README and
the CHANGELOG), and these tests fail the build when any of them drifts, which
is the failure mode that produced a v0.7 copy of the module sitting beside a
v0.7.7 package.

Everything that inspects pypi_staging/ skips when that directory is absent: it
is deliberately missing from the tree published to GitHub.
"""
import ast
import io
import re
from pathlib import Path

import pytest

import datatypical
import datatypical_viz
from datatypical import DataTypical

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "pypi_staging"

EXPECTED = "0.8.3"


def _read(path):
    return io.open(path, encoding="utf-8").read()


def _leading_changelog_entry():
    """The CHANGELOG text from its first version heading up to the second."""
    text = _read(ROOT / "CHANGELOG.md")
    starts = [m.start() for m in re.finditer(r"^## \[", text, flags=re.MULTILINE)]
    assert starts, "CHANGELOG has no version headings"
    return text[starts[0]:starts[1] if len(starts) > 1 else len(text)]


class TestVersionConsistency:
    def test_the_single_source(self):
        assert datatypical.__version__ == EXPECTED

    def test_module_docstring(self):
        first_lines = datatypical.__doc__.splitlines()[:3]
        assert any(EXPECTED in line for line in first_lines), first_lines

    def test_viz_module_docstring(self):
        first_lines = datatypical_viz.__doc__.splitlines()[:3]
        assert any(EXPECTED in line for line in first_lines), first_lines

    def test_to_config(self):
        assert DataTypical().to_config()["version"] == datatypical.__version__

    def test_to_config_reads_the_single_source(self):
        """v0.8.2: no second copy of the version is typed into to_config."""
        assert re.search(r'cfg\["version"\] = __version__', _read(ROOT / "datatypical.py"))
        assert not re.search(r'cfg\["version"\] = "', _read(ROOT / "datatypical.py"))

    def test_setup_py_reads_the_single_source(self):
        text = _read(ROOT / "setup.py")
        assert "version=_version()" in text
        assert not re.search(r'version="[\d.]+"', text)
        # the expression setup.py uses must find the same version
        found = re.search(r'^__version__ = "([^"]+)"', _read(ROOT / "datatypical.py"),
                          re.MULTILINE)
        assert found and found.group(1) == datatypical.__version__

    def test_the_version_is_a_plain_literal(self):
        """setuptools reads the attr statically, which needs a string literal."""
        tree = ast.parse(_read(ROOT / "datatypical.py"))
        values = [node.value.value for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", None) == "__version__" for t in node.targets)
                  and isinstance(node.value, ast.Constant)]
        assert values == [EXPECTED]

    def test_changelog_leads_with_this_version(self):
        text = _read(ROOT / "CHANGELOG.md")
        headings = re.findall(r"^## \[([^\]]+)\]", text, flags=re.MULTILINE)
        assert headings, "CHANGELOG has no version headings"
        assert headings[0] == EXPECTED, headings[:3]

    def test_readme_reports_the_current_version(self):
        text = _read(ROOT / "README.md")
        assert "**Current Version**: %s" % EXPECTED in text

    def test_citation_version(self):
        text = _read(ROOT / "README.md")
        assert "version = {%s}" % EXPECTED in text

    @pytest.mark.skipif(not STAGING.exists(), reason="pypi_staging not present")
    def test_staging_pyproject_reads_the_single_source(self):
        text = _read(STAGING / "pyproject.toml")
        assert 'dynamic = ["version"]' in text
        assert 'version = {attr = "datatypical.__version__"}' in text
        assert not re.search(r'^version = "', text, flags=re.MULTILINE)


class TestDeclaredDependencies:
    def test_py_pcha_is_declared_in_requirements(self):
        """archetypal_method='aa' needs it, so it cannot stay undeclared."""
        text = _read(ROOT / "requirements.txt")
        assert "py_pcha" in text

    @pytest.mark.skipif(not STAGING.exists(), reason="pypi_staging not present")
    def test_py_pcha_is_declared_in_the_staging_metadata(self):
        text = _read(STAGING / "pyproject.toml")
        assert "py_pcha" in text

    def test_every_requirement_is_importable_or_optional(self):
        """Guards against a typo in a pinned name."""
        text = _read(ROOT / "requirements.txt")
        names = [re.split(r"[<>=!]", line)[0].strip()
                 for line in text.splitlines()
                 if line.strip() and not line.startswith("#")]
        assert "py_pcha" in names
        assert "numpy" in names
        assert "scikit-learn" in names

    # The distribution name of each third-party package the modules import.
    DISTRIBUTION = {"sklearn": "scikit-learn", "scipy": "scipy", "numpy": "numpy",
                    "pandas": "pandas", "numba": "numba", "networkx": "networkx",
                    "py_pcha": "py_pcha", "joblib": "joblib",
                    "threadpoolctl": "threadpoolctl", "matplotlib": "matplotlib",
                    "seaborn": "seaborn"}

    @staticmethod
    def _imported_roots(path):
        roots = set()
        for node in ast.walk(ast.parse(_read(path))):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        return roots

    def _declared(self, text):
        return {re.split(r"[<>=!\s\"]", line.strip().strip('",'))[0].lower()
                for line in text.splitlines() if line.strip()}

    def test_every_third_party_import_is_declared(self):
        """
        v0.8.2: joblib and threadpoolctl are imported unguarded at module top
        and were declared nowhere, so installs worked only because scikit-learn
        happens to pull both in.
        """
        imported = (self._imported_roots(ROOT / "datatypical.py")
                    | self._imported_roots(ROOT / "datatypical_viz.py"))
        third_party = {r for r in imported if r in self.DISTRIBUTION}
        unknown = {r for r in imported
                   if r not in self.DISTRIBUTION and r not in _STDLIB}
        assert not unknown, "imports with no known distribution: %s" % unknown
        declared = self._declared(_read(ROOT / "requirements.txt"))
        missing = {self.DISTRIBUTION[r] for r in third_party} - declared
        assert not missing, "imported but not in requirements.txt: %s" % missing

    @pytest.mark.skipif(not STAGING.exists(), reason="pypi_staging not present")
    def test_the_staging_metadata_declares_the_same_set(self):
        text = _read(STAGING / "pyproject.toml")
        block = text[text.index("dependencies = ["):]
        block = block[:block.index("]")]
        declared = {re.split(r"[<>=!]", item.strip().strip('",'))[0].lower()
                    for item in block.splitlines()[1:] if item.strip().strip('",')}
        required = self._declared(_read(ROOT / "requirements.txt"))
        assert declared == required


_STDLIB = {"__future__", "dataclasses", "typing", "heapq", "math", "gc", "os",
           "warnings", "hashlib", "numbers", "itertools", "functools", "re",
           "collections", "sys"}


@pytest.mark.skipif(not STAGING.exists(), reason="pypi_staging not present")
class TestStagingParity:
    @pytest.mark.parametrize(
        "name", ["datatypical.py", "datatypical_viz.py", "LICENSE",
                 "requirements.txt", "CHANGELOG.md"],
    )
    def test_staging_copy_matches_the_root(self, name):
        source, target = ROOT / name, STAGING / name
        assert target.exists(), "%s missing from pypi_staging" % name
        assert source.read_bytes() == target.read_bytes(), (
            "%s has drifted between the project root and pypi_staging; "
            "run pypi_staging/prepare_staging.py" % name
        )

    def test_the_pypi_readme_carries_no_badges(self):
        """Badges link relatively; those links break on a PyPI project page."""
        text = _read(STAGING / "README.md")
        assert "img.shields.io" not in text
        assert "badge.fury.io" not in text

    def test_the_repo_readme_does_carry_badges(self):
        text = _read(ROOT / "README.md")
        assert "img.shields.io" in text


class TestPrivateDocumentsAreNotPublished:
    """
    Internal process records stay private. The filename is assembled at runtime
    so this guard does not trip over its own source.
    """

    @staticmethod
    def _private_names():
        return ["DATATYPICAL" + "_ISSUES.md", "BUG" + "_SWEEPS.md",
                "RELEASING" + ".md", "DATATYPICAL" + "_PLAN.md"]

    def test_no_shipped_file_mentions_a_private_document(self):
        shipped = [ROOT / "datatypical.py", ROOT / "datatypical_viz.py",
                   ROOT / "README.md", ROOT / "CHANGELOG.md",
                   ROOT / "setup.py", ROOT / "requirements.txt"]
        offenders = []
        for path in shipped:
            if not path.exists():
                continue
            text = _read(path)
            for name in self._private_names() + ["PLAN" + "_0_8", "DESIGN" + "_stereo"]:
                if name in text:
                    offenders.append("%s mentions %s" % (path.name, name))
        assert not offenders, offenders

    def test_no_private_document_sits_in_the_project_root(self):
        """They live one level up, outside the tree that gets published."""
        present = [n for n in self._private_names() if (ROOT / n).exists()]
        assert not present, (
            "private documents found inside the publishable tree: %s" % present
        )


class TestDocumentationCoherence:
    """
    After a long run of edits the docs drift apart before anyone notices. These
    checks caught a stale test count and a public function that was exported and
    changelogged but never mentioned in the README.
    """

    def test_the_version_agrees_everywhere(self):
        sources = {
            "__version__": re.findall(r'^__version__ = "([\d.]+)"',
                                      _read(ROOT / "datatypical.py"), re.M),
            "module docstring": re.findall(r"DataTypical v([\d.]+) ---",
                                           _read(ROOT / "datatypical.py")),
            "viz docstring": re.findall(r"DataTypical v([\d.]+) - Visualization",
                                        _read(ROOT / "datatypical_viz.py")),
            "README citation": re.findall(r"version = \{([\d.]+)\}",
                                          _read(ROOT / "README.md")),
            "README current": re.findall(r"\*\*Current Version\*\*: ([\d.]+)",
                                         _read(ROOT / "README.md")),
            "CHANGELOG": re.findall(r"^## \[([\d.]+)\]",
                                    _read(ROOT / "CHANGELOG.md"), re.M)[:1],
        }
        assert all(sources.values()), "a version source went missing: %s" % sources
        found = {v for vals in sources.values() for v in vals}
        assert found == {EXPECTED}, "version disagrees: %s" % sources

    def test_the_test_count_agrees_across_the_docs(self):
        """The badge, the README text and the leading CHANGELOG entry."""
        readme = _read(ROOT / "README.md")
        counts = {
            "badge": re.findall(r"tests-(\d+)%20passing", readme),
            "README": re.findall(r"pytest suite: (\d+) tests", readme),
            "CHANGELOG": re.findall(r"The suite has (\d+) tests", _leading_changelog_entry()),
        }
        assert all(counts.values()), "a test count went missing: %s" % counts
        found = {c for vals in counts.values() for c in vals}
        assert len(found) == 1, "test count disagrees across docs: %s" % counts

    def test_the_coverage_figure_agrees_across_the_docs(self):
        """
        v0.8.2: the sibling of the test-count guard. The figure moved by six
        points with whether Numba was compiling, which is a property of the
        machine, not the code, so the convention is stated beside the number.
        """
        readme = _read(ROOT / "README.md")
        entry = _leading_changelog_entry()
        figures = {
            "badge": re.findall(r"coverage-([\d.]+)%25", readme),
            "README": re.findall(r"tests, ([\d.]+)% statement coverage", readme),
            "CHANGELOG": re.findall(r"tests, ([\d.]+)% statement coverage", entry),
        }
        assert all(figures.values()), "a coverage figure went missing: %s" % figures
        found = {f for vals in figures.values() for f in vals}
        assert len(found) == 1, "coverage figure disagrees across docs: %s" % figures
        for name, text in (("README", readme), ("CHANGELOG", entry)):
            assert "NUMBA_DISABLE_JIT" in text, "%s states no convention" % name

    def test_every_readme_anchor_resolves(self):
        readme = _read(ROOT / "README.md")
        anchors = set()
        for line in readme.splitlines():
            m = re.match(r"^#{2,4}\s+(.*)$", line)
            if m:
                slug = re.sub(r"[^\w\s-]", "", m.group(1).strip().lower()).strip()
                anchors.add(re.sub(r"\s+", "-", slug))
        broken = [t for _, t in re.findall(r"\[([^\]]+)\]\((#[^)]+)\)", readme)
                  if t[1:] not in anchors]
        assert not broken, "README links do not resolve: %s" % broken

    def test_every_relative_readme_link_exists(self):
        """A link into a folder the GitHub tree does not ship is a dead link."""
        readme = _read(ROOT / "README.md")
        targets = [t for _, t in re.findall(r"\[([^\]]+)\]\(([^)#][^)]*)\)", readme)
                   if not re.match(r"[a-z]+://", t)]
        missing = [t for t in targets if not (ROOT / t.split("#")[0]).exists()]
        assert not missing, "README links to missing paths: %s" % missing
        unshipped = [t for t in targets
                     if t.split("/")[0].rstrip("/") in {"DRAFTS", "prototypes", "pypi_staging"}]
        assert not unshipped, "README links into folders that are not published: %s" % unshipped

    @pytest.mark.parametrize(
        "name",
        ["formative_method", "exact_formative_archetypal",
         "exact_formative_prototypical", "exact_formative_stereotypical",
         "explanation_method", "exact_explanations_archetypal",
         "exact_explanations_prototypical", "exact_explanations_stereotypical",
         "split_half_rho", "archetypal_backend_", "archetypal_method",
         "__version__", "NUMBA_AVAILABLE", "plot_size"],
    )
    def test_public_surface_is_documented(self, name):
        """Anything the code exposes should be findable in the README."""
        code = _read(ROOT / "datatypical.py") + _read(ROOT / "datatypical_viz.py")
        assert name in code, "%s not in the code" % name
        assert name in _read(ROOT / "README.md"), \
            "%s is public but the README never mentions it" % name

    @pytest.mark.parametrize(
        "relative",
        ["CHANGELOG.md", "README.md", "docs/COMPUTATION_GUIDE.md", "docs/EXAMPLES.md",
         "docs/INTERPRETATION_GUIDE.md", "docs/PACKAGE_SUMMARY.md",
         "docs/QUICK_REFERENCE.md", "docs/START_HERE.md",
         "docs/VISUALIZATION_GUIDE.md", "tests/TEST_SUITE_GUIDE.md",
         "benchmarks/README.md", "datatypical.py", "datatypical_viz.py"],
    )
    def test_no_em_dashes(self, relative):
        path = ROOT / relative
        if not path.exists():
            pytest.skip("%s not present" % relative)
        text = _read(path)
        assert text.count("—") == 0, "%s contains an em dash" % relative

    def test_the_changelog_carries_no_en_dashes(self):
        text = _read(ROOT / "CHANGELOG.md")
        assert text.count("–") == 0
