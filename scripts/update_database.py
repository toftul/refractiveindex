"""Replace refractiveindex/database/ with a release of refractiveindex.info-database.

Usage:
    python scripts/update_database.py [TAG] [--changelog]

TAG is an upstream release tag, for example v2026-05-24. Without TAG, the script uses the
latest release. The script copies the database/ folder and the LICENSE file of that release
into refractiveindex/database/ and removes all other files there. Then it writes the tag and
its commit to refractiveindex/_database_version.py. A second run for the same tag changes
nothing.

With --changelog, the script also adds a line about the update to CHANGELOG.md under
[Unreleased]. Set GITHUB_TOKEN to get a higher GitHub API rate limit.
"""

import argparse
import io
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM = "polyanskiy/refractiveindex.info-database"
ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = ROOT / "refractiveindex" / "database"
VERSION_FILE = ROOT / "refractiveindex" / "_database_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"

# Changelog subsections that come after "### Changed" (https://keepachangelog.com/en/1.1.0/).
_SECTIONS_AFTER_CHANGED = ("### Deprecated", "### Removed", "### Fixed", "### Security")


def _get(url):
    request = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def latest_tag():
    """Return the tag of the latest upstream release."""
    return json.loads(_get(f"https://api.github.com/repos/{UPSTREAM}/releases/latest"))["tag_name"]


def resolve_commit(tag):
    """Return the commit SHA that an upstream tag points to."""
    url = f"https://api.github.com/repos/{UPSTREAM}/commits/{urllib.parse.quote(tag, safe='')}"
    return json.loads(_get(url))["sha"]


def download_archive(commit):
    """Return the GitHub zip archive of an upstream commit as bytes."""
    return _get(f"https://github.com/{UPSTREAM}/archive/{commit}.zip")


def extract_database(archive, dest):
    """Write database/* and LICENSE from a GitHub zip archive into dest. Return the file count."""
    count = 0
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Member names are "<repo>-<commit>/<path in the repo>".
            _, _, path = info.filename.partition("/")
            if path == "LICENSE":
                rel = Path("LICENSE")
            elif path.startswith("database/"):
                rel = Path(path[len("database/"):])
            else:
                continue
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe path in archive: {info.filename!r}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            count += 1
    return count


def write_version_file(tag, commit):
    VERSION_FILE.write_text(
        "# Version of the refractiveindex.info database in refractiveindex/database/.\n"
        "# Written by scripts/update_database.py. Do not edit.\n"
        f'DATABASE_VERSION = "{tag}"\n'
        f'DATABASE_COMMIT = "{commit}"\n',
        encoding="utf-8",
    )


def update(tag):
    """Replace DATABASE_DIR with the database of an upstream tag. Return (commit, file count)."""
    commit = resolve_commit(tag)
    archive = download_archive(commit)
    with tempfile.TemporaryDirectory(dir=DATABASE_DIR.parent) as tempdir:
        new = Path(tempdir) / "database"
        count = extract_database(archive, new)
        if not (new / "catalog-nk.yml").is_file():
            raise RuntimeError(f"{tag} has no database/catalog-nk.yml. Did the upstream layout change?")
        if DATABASE_DIR.exists():
            shutil.rmtree(DATABASE_DIR)
        new.rename(DATABASE_DIR)
    write_version_file(tag, commit)
    return commit, count


def add_changelog_entry(tag, changelog=CHANGELOG):
    """Add a line about the database update under [Unreleased] -> Changed in the changelog."""
    entry = f"- The included database is now {tag} of refractiveindex.info-database.\n"
    lines = changelog.read_text(encoding="utf-8").splitlines(keepends=True)

    releases = [i for i, line in enumerate(lines) if line.startswith("## [")]
    if not releases:
        raise ValueError(f"No release section in {changelog}")
    start = releases[0]
    if not lines[start].startswith("## [Unreleased]"):
        lines[start:start] = ["## [Unreleased]\n", "\n"]
        releases = [start] + [i + 2 for i in releases]
    end = releases[1] if len(releases) > 1 else len(lines)
    section = lines[start:end]

    if entry in section:
        return
    if "### Changed\n" in section:
        # First item of the existing "Changed" list.
        i = start + section.index("### Changed\n") + 1
        while i < end and lines[i].strip() == "":
            i += 1
        lines[i:i] = [entry]
    else:
        later = [j for j, line in enumerate(section) if line.startswith(_SECTIONS_AFTER_CHANGED)]
        i = start + later[0] if later else end
        lines[i:i] = ["### Changed\n", "\n", entry, "\n"]
    changelog.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", nargs="?", help="upstream release tag, for example v2026-05-24 (default: the latest release)")
    parser.add_argument("--changelog", action="store_true", help="add a line about the update to CHANGELOG.md")
    args = parser.parse_args()

    tag = args.tag or latest_tag()
    commit, count = update(tag)
    if args.changelog:
        add_changelog_entry(tag)
    print(f"refractiveindex/database/ is now {tag} (commit {commit}, {count} files)")


if __name__ == "__main__":
    main()
