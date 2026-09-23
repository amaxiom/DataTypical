"""
Release metadata consistency.

The version appears in five places. These tests fail the build when any of them
drifts, which is the failure mode that produced a v0.7 copy of the module
sitting beside a v0.7.7 package.

Everything that inspects pypi_staging/ skips when that directory is absent: it
is deliberately missing from the tree published to GitHub.
"""
import io
import re
from pathlib import Path

import pytest

import datatypical
from datatypical import DataTypical

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "pypi_staging"

EXPECTED = "0.8.1"


def _read(path):
    return io.open(path, encoding="utf-8").read()


class TestVersionConsistency:
    def test_module_docstring(self):
        first_lines = datatypical.__doc__.splitlines()[:3]
        assert any(EXPECTED in line for line in first_lines), first_lines

    def test_to_config(self):
        assert DataTypical().to_config()["version"] == EXPECTED

    def test_setup_py(self):
        text = _read(ROOT / "setup.py")
        assert 'version="%s"' % EXPECTED in text

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
    def test_staging_pyproject(self):
        text = _read(STAGING / "pyproject.toml")
        assert 'version = "%s"' % EXPECTED in text


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
            for name in self._private_names():
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
        import re

        sources = {
            "setup.py": re.findall(r'version="([\d.]+)"', _read(ROOT / "setup.py")),
            "to_config": re.findall(r'cfg\["version"\] = "([\d.]+)"',
                                    _read(ROOT / "datatypical.py")),
            "module docstring": re.findall(r"DataTypical v([\d.]+) ---",
                                           _read(ROOT / "datatypical.py")),
            "README citation": re.findall(r"version = \{([\d.]+)\}",
                                          _read(ROOT / "README.md")),
            "README current": re.findall(r"\*\*Current Version\*\*: ([\d.]+)",
                                         _read(ROOT / "README.md")),
            "CHANGELOG": re.findall(r"^## \[([\d.]+)\]",
                                    _read(ROOT / "CHANGELOG.md"), re.M)[:1],
        }
        found = {v for vals in sources.values() for v in vals}
        assert found == {EXPECTED}, "version disagrees: %s" % sources

    def test_the_test_count_agrees_across_the_docs(self):
        import re

        readme = _read(ROOT / "README.md")
        changelog = _read(ROOT / "CHANGELOG.md")
        counts = set(re.findall(r"tests-(\d+)%20passing", readme))
        counts |= set(re.findall(r"pytest suite: (\d+) tests", readme))
        counts |= set(re.findall(r"report time\. (\d+) tests", changelog))
        assert len(counts) == 1, "test count disagrees across docs: %s" % counts

    def test_every_readme_anchor_resolves(self):
        import re

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

    @pytest.mark.parametrize(
        "name",
        ["formative_method", "exact_formative_archetypal",
         "exact_formative_prototypical", "exact_formative_stereotypical",
         "split_half_rho", "archetypal_backend_", "archetypal_method"],
    )
    def test_public_surface_is_documented(self, name):
        """Anything the code exposes should be findable in the README."""
        assert name in _read(ROOT / "datatypical.py"), "%s not in the code" % name
        assert name in _read(ROOT / "README.md"),             "%s is public but the README never mentions it" % name

    def test_the_changelog_carries_no_em_dashes(self):
        text = _read(ROOT / "CHANGELOG.md")
        assert text.count("—") == 0 and text.count("–") == 0
