"""Tests for scripts/update_database.py. No network is used."""

import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "update_database.py"
_spec = importlib.util.spec_from_file_location("update_database", _SCRIPT)
update_database = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_database)


def _make_archive(files):
    """Return a zip archive (bytes) laid out like a GitHub archive of the upstream repo."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("refractiveindex.info-database-abc123/", "")
        for name, content in files.items():
            zf.writestr(f"refractiveindex.info-database-abc123/{name}", content)
    return buffer.getvalue()


_UPSTREAM_FILES = {
    "README.md": "upstream readme",
    "LICENSE": "CC0",
    "tests/test_yaml.py": "",
    "database/catalog-nk.yml": "catalog",
    "database/data/main/X/nk/Y.yml": "data",
}


class TestExtractDatabase(unittest.TestCase):
    """extract_database copies only database/* and LICENSE."""

    def test_extract(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dest = Path(tempdir) / "database"
            count = update_database.extract_database(_make_archive(_UPSTREAM_FILES), dest)
            files = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
            self.assertEqual(files, {"LICENSE", "catalog-nk.yml", "data/main/X/nk/Y.yml"})
            self.assertEqual(count, 3)
            self.assertEqual((dest / "data/main/X/nk/Y.yml").read_text(), "data")

    def test_unsafe_path(self):
        """A member that points outside dest raises ValueError."""
        archive = _make_archive({"database/../../evil.yml": "x"})
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ValueError):
                update_database.extract_database(archive, Path(tempdir) / "database")
            self.assertFalse((Path(tempdir) / "evil.yml").exists())


class TestUpdate(unittest.TestCase):
    """update replaces the whole folder and writes the version file."""

    def test_update(self):
        with tempfile.TemporaryDirectory() as tempdir:
            database_dir = Path(tempdir) / "database"
            version_file = Path(tempdir) / "_database_version.py"
            (database_dir / "data").mkdir(parents=True)
            (database_dir / "data" / "removed-upstream.yml").write_text("old")
            with mock.patch.object(update_database, "DATABASE_DIR", database_dir), \
                    mock.patch.object(update_database, "VERSION_FILE", version_file), \
                    mock.patch.object(update_database, "resolve_commit", return_value="abc123"), \
                    mock.patch.object(update_database, "download_archive",
                                      return_value=_make_archive(_UPSTREAM_FILES)):
                self.assertEqual(update_database.update("v2026-05-24"), ("abc123", 3))

            self.assertFalse((database_dir / "data" / "removed-upstream.yml").exists())
            self.assertTrue((database_dir / "catalog-nk.yml").is_file())
            version = {}
            exec(version_file.read_text(), version)
            self.assertEqual(version["DATABASE_VERSION"], "v2026-05-24")
            self.assertEqual(version["DATABASE_COMMIT"], "abc123")

    def test_missing_catalog(self):
        """An archive without database/catalog-nk.yml raises and keeps the old folder."""
        with tempfile.TemporaryDirectory() as tempdir:
            database_dir = Path(tempdir) / "database"
            database_dir.mkdir()
            (database_dir / "catalog-nk.yml").write_text("old")
            with mock.patch.object(update_database, "DATABASE_DIR", database_dir), \
                    mock.patch.object(update_database, "VERSION_FILE", Path(tempdir) / "v.py"), \
                    mock.patch.object(update_database, "resolve_commit", return_value="abc123"), \
                    mock.patch.object(update_database, "download_archive",
                                      return_value=_make_archive({"LICENSE": "CC0"})):
                with self.assertRaises(RuntimeError):
                    update_database.update("v2026-05-24")
            self.assertEqual((database_dir / "catalog-nk.yml").read_text(), "old")


_HEADER = "# Changelog\n\nIntro.\n\n"
_RELEASE = "## [1.2.0] - 2026-10-26\n\n### Fixed\n\n- A fix.\n\n[1.2.0]: https://example.com\n"
_ENTRY = "- The included database is now v2027-01-01 of refractiveindex.info-database.\n"


class TestAddChangelogEntry(unittest.TestCase):
    """add_changelog_entry puts the line under [Unreleased] -> Changed."""

    def _run(self, text):
        with tempfile.TemporaryDirectory() as tempdir:
            changelog = Path(tempdir) / "CHANGELOG.md"
            changelog.write_text(text)
            update_database.add_changelog_entry("v2027-01-01", changelog)
            return changelog.read_text()

    def test_no_unreleased_section(self):
        self.assertEqual(
            self._run(_HEADER + _RELEASE),
            _HEADER + "## [Unreleased]\n\n### Changed\n\n" + _ENTRY + "\n" + _RELEASE,
        )

    def test_changed_goes_between_added_and_fixed(self):
        unreleased = "## [Unreleased]\n\n### Added\n\n- New.\n\n### Fixed\n\n- Fix.\n\n"
        self.assertEqual(
            self._run(_HEADER + unreleased + _RELEASE),
            _HEADER + "## [Unreleased]\n\n### Added\n\n- New.\n\n"
            + "### Changed\n\n" + _ENTRY + "\n"
            + "### Fixed\n\n- Fix.\n\n" + _RELEASE,
        )

    def test_existing_changed_list(self):
        unreleased = "## [Unreleased]\n\n### Changed\n\n- Other change.\n\n"
        self.assertEqual(
            self._run(_HEADER + unreleased + _RELEASE),
            _HEADER + "## [Unreleased]\n\n### Changed\n\n" + _ENTRY + "- Other change.\n\n" + _RELEASE,
        )

    def test_second_run_changes_nothing(self):
        once = self._run(_HEADER + _RELEASE)
        self.assertEqual(self._run(once), once)
