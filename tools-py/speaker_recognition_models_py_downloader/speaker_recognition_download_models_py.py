#!/usr/bin/env python3
"""Download pinned upstream Speaker Recognition model artifacts for local model conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

CATALOG_RELATIVE = Path("tools-py") / "speaker_recognition_models_py_downloader" / "catalog.json"
MODEL_ROOT_RELATIVE = Path("models") / "py"
PROVIDERS = {"huggingface", "github_release", "pypi_package"}


class DownloadError(Exception):
    """Expected catalog, provider, or file-safety error."""


def _relative(value, label, allow_dot=False):
    if allow_dot and value == ".":
        return PurePosixPath(".")
    if not isinstance(value, str) or not value.strip():
        raise DownloadError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or "\x00" in value
        or any(
            part.endswith((".", " "))
            or any(character in '<>:"|?*' or ord(character) < 32 for character in part)
            for part in path.parts
        )
    ):
        raise DownloadError(f"{label} must stay within its configured directory: {value!r}")
    return path


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise DownloadError(f"Cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise DownloadError(f"Expected a JSON object in {path}")
    return value


def _load_project(root):
    root = root.resolve()
    catalog_path = root / CATALOG_RELATIVE
    if (
        catalog_path.is_symlink()
        or not catalog_path.is_file()
        or not catalog_path.resolve().is_relative_to(root)
    ):
        raise DownloadError(f"Project model catalog not found inside the project: {catalog_path}")
    catalog = _read_json(catalog_path)
    if catalog.get("schema_version") != 1:
        raise DownloadError("Catalog schema_version must be 1")
    project_id = catalog.get("project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        raise DownloadError("Catalog project_id must be a non-empty string")
    models_root = (root / MODEL_ROOT_RELATIVE).resolve()
    if not models_root.is_relative_to(root):
        raise DownloadError(f"Model output directory must stay inside the project: {models_root}")
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list):
        raise DownloadError("Catalog artifacts must be an array")
    ids = set()
    file_ids = set()
    for index, artifact in enumerate(artifacts):
        field = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            raise DownloadError(f"{field} must be an object")
        artifact_id = artifact.get("id")
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in artifact_id)
        ):
            raise DownloadError(f"{field}.id contains unsupported characters")
        if artifact_id in ids:
            raise DownloadError(f"Duplicate artifact id: {artifact_id}")
        ids.add(artifact_id)
        _relative(artifact.get("destination"), f"{field}.destination", allow_dot=True)
        source = artifact.get("source")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("provider"), str)
            or source["provider"] not in PROVIDERS
        ):
            raise DownloadError(f"{field}.source.provider must be one of {sorted(PROVIDERS)}")
        _validate_source(source, field)
        entries = source["assets"] if source["provider"] == "github_release" else source["files"]
        for file_entry in entries:
            file_id = file_entry.get("id")
            if file_id is not None:
                if file_id in file_ids:
                    raise DownloadError(f"Duplicate model file id: {file_id}")
                file_ids.add(file_id)
    all_targets = set()
    for artifact in artifacts:
        dest = _relative(artifact["destination"], "artifact.destination", allow_dot=True)
        source = artifact["source"]
        entries = source["assets"] if source["provider"] == "github_release" else source["files"]
        for entry in entries:
            key = "asset" if source["provider"] == "github_release" else "path"
            target = dest / _relative(entry.get("destination", PurePosixPath(entry[key]).name), "entry.destination")
            norm = tuple(part.casefold() for part in target.parts)
            if norm in all_targets:
                raise DownloadError(f"Duplicate destination file across artifacts: {target}")
            all_targets.add(norm)
    return catalog, models_root


def _validate_source(source, field):
    provider = source["provider"]
    if provider == "huggingface":
        if not isinstance(source.get("repo_id"), str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
            source["repo_id"],
        ):
            raise DownloadError(f"{field}.source.repo_id must use owner/name format")
        if not isinstance(source.get("revision"), str) or not source["revision"].strip():
            raise DownloadError(f"{field}.source.revision must be pinned explicitly")
        entries = source.get("files")
        key = "path"
    elif provider == "github_release":
        if not isinstance(source.get("repository"), str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
            source["repository"],
        ):
            raise DownloadError(f"{field}.source.repository must use owner/name format")
        if not isinstance(source.get("tag"), str) or not source["tag"].strip():
            raise DownloadError(f"{field}.source.tag must be pinned explicitly")
        entries = source.get("assets")
        key = "asset"
    else:
        package = source.get("package")
        version = source.get("version")
        if not isinstance(package, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", package):
            raise DownloadError(f"{field}.source.package must be set")
        if not isinstance(version, str) or not re.fullmatch(
            r"\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
            version,
        ):
            raise DownloadError(f"{field}.source.version must be an exact version")
        wheel_sha256 = source.get("wheel_sha256")
        if (
            not isinstance(wheel_sha256, str)
            or len(wheel_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in wheel_sha256)
        ):
            raise DownloadError(f"{field}.source.wheel_sha256 must be 64 hex characters")
        entries = source.get("files")
        key = "path"
    if not isinstance(entries, list) or not entries:
        raise DownloadError(f"{field}.source must explicitly select one or more files")
    destinations = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise DownloadError(f"{field}.source file entry {index} must be an object")
        _relative(entry.get(key), f"{field}.source file entry {index}.{key}")
        target = _relative(
            entry.get("destination", PurePosixPath(entry[key]).name),
            f"{field}.source file entry {index}.destination",
        )
        if target.name.casefold().startswith(".manifest-") or target.as_posix().casefold() == "artifact-manifest.json":
            raise DownloadError("Manifest filenames are reserved for download provenance")
        target_parts = tuple(part.casefold() for part in target.parts)
        if any(
            target_parts[:len(existing)] == existing
            or existing[:len(target_parts)] == target_parts
            for existing in destinations
        ):
            raise DownloadError(f"Overlapping output paths in {field}: {target}")
        destinations.add(target_parts)
        expected = entry.get("sha256")
        if expected is not None and (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
        ):
            raise DownloadError(f"{field}.source file entry {index}.sha256 must be 64 hex characters")
        file_id = entry.get("id")
        if file_id is not None and (
            not isinstance(file_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", file_id)
        ):
            raise DownloadError(f"{field}.source file entry {index}.id is invalid")
        if "default" in entry and not isinstance(entry["default"], bool):
            raise DownloadError(f"{field}.source file entry {index}.default must be a boolean")
        if "default" in entry and file_id is None:
            raise DownloadError(f"{field}.source file entry {index} needs an id when default is specified")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path, expected, label):
    actual = _sha256(path)
    if expected and actual.casefold() != expected.casefold():
        raise DownloadError(f"SHA-256 mismatch for {label}: expected {expected}, received {actual}")
    return actual


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(request, response, code, message, headers, new_url)
        if redirected is not None and urllib.parse.urlsplit(request.full_url).hostname != urllib.parse.urlsplit(new_url).hostname:
            redirected.remove_header("Authorization")
        return redirected


def _request(url, output, headers=None):
    opener = urllib.request.build_opener(_SafeRedirect)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "esp-idf-speaker-recognition-model-downloader/1", **(headers or {})},
    )
    try:
        with opener.open(request, timeout=90) as response, output.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        raise DownloadError(f"Download failed from {url}: {error}") from error


def _download_huggingface(source, entries, stage):
    owner, repo = source["repo_id"].split("/", 1)
    token = os.environ.get("HF_TOKEN")
    downloaded = []
    for entry in entries:
        remote = _relative(entry["path"], "Hugging Face path").as_posix()
        target = _relative(entry.get("destination", PurePosixPath(remote).name), "Hugging Face output path")
        revision = urllib.parse.quote(source["revision"], safe="")
        quoted_repo = urllib.parse.quote(f"{owner}/{repo}", safe="/")
        quoted_path = urllib.parse.quote(remote, safe="/")
        url = f"https://huggingface.co/{quoted_repo}/resolve/{revision}/{quoted_path}"
        staged = stage.joinpath(*target.parts)
        staged.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {token}"} if token else None
        _request(url, staged, headers)
        downloaded.append((entry, target, staged, f"huggingface:{source['repo_id']}@{source['revision']}:{remote}"))
    return downloaded


def _download_github(source, entries, stage):
    owner, repo = source["repository"].split("/", 1)
    tag = urllib.parse.quote(source["tag"], safe="")
    metadata_path = stage / ".release-metadata.json"
    _request(f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}", metadata_path,
             {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    release = _read_json(metadata_path)
    metadata_path.unlink()
    if release.get("tag_name") != source["tag"] or not isinstance(release.get("assets"), list):
        raise DownloadError(f"GitHub did not return the requested release {source['repository']}@{source['tag']}")
    output = []
    for entry in entries:
        matches = [asset for asset in release["assets"] if asset.get("name") == entry["asset"]]
        if len(matches) != 1:
            raise DownloadError(f"Expected one release asset {entry['asset']!r}; found {len(matches)}")
        url = matches[0].get("browser_download_url")
        prefix = f"https://github.com/{owner}/{repo}/releases/download/"
        if not isinstance(url, str) or not url.startswith(prefix):
            raise DownloadError(f"Unexpected URL for GitHub asset {entry['asset']!r}")
        target = _relative(entry.get("destination", entry["asset"]), "GitHub output path")
        staged = stage.joinpath(*target.parts)
        staged.parent.mkdir(parents=True, exist_ok=True)
        _request(url, staged)
        output.append((entry, target, staged, f"github_release:{source['repository']}@{source['tag']}:{entry['asset']}"))
    return output


def _download_pypi(source, entries, stage):
    wheel_dir = stage / ".wheel"
    payload_dir = stage / ".payload"
    wheel_dir.mkdir()
    payload_dir.mkdir()
    command = [
        sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
        "--disable-pip-version-check", "--dest", str(wheel_dir),
        f"{source['package']}=={source['version']}",
    ]
    try:
        subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PIP_NO_INPUT": "1"},
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise DownloadError(f"Could not download {source['package']}=={source['version']}: {detail}") from error
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DownloadError(f"Could not download {source['package']}=={source['version']}: {error}") from error
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise DownloadError(f"Expected one compatible wheel; found {len(wheels)}")
    wheel_hash = _verify(
        wheels[0],
        source["wheel_sha256"],
        f"{source['package']}=={source['version']} wheel",
    )
    output = []
    try:
        with zipfile.ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())
            for entry in entries:
                member = _relative(entry["path"], "PyPI wheel member").as_posix()
                if member not in names:
                    raise DownloadError(f"Selected wheel member not found: {member}")
                target = _relative(entry.get("destination", PurePosixPath(member).name), "PyPI output path")
                staged = payload_dir.joinpath(*target.parts)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as member_stream, staged.open("wb") as destination:
                    shutil.copyfileobj(member_stream, destination, length=1024 * 1024)
                output.append((
                    entry, target, staged,
                    f"pypi:{source['package']}=={source['version']}:{member}",
                    wheels[0].name, wheel_hash,
                ))
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise DownloadError(f"Could not inspect PyPI wheel {wheels[0].name}: {error}") from error
    return output

def _download_artifact(artifact, models_root, overwrite):
    artifact_id = artifact["id"]
    destination = _relative(artifact["destination"], f"{artifact_id}.destination", allow_dot=True)
    final_directory = models_root.joinpath(*destination.parts)
    if not final_directory.resolve().is_relative_to(models_root.resolve()):
        raise DownloadError(f"Artifact output escapes models_root: {final_directory}")
    source = artifact["source"]
    manifest_source = artifact.get("_manifest_source", source)
    stage_parent = models_root
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{artifact_id}-stage-", dir=stage_parent) as temp:
        stage = Path(temp)
        if source["provider"] == "huggingface":
            files = _download_huggingface(source, source["files"], stage)
        elif source["provider"] == "github_release":
            files = _download_github(source, source["assets"], stage)
        else:
            files = _download_pypi(source, source["files"], stage)
        enriched = []
        targets = []
        for row in files:
            entry, relative, staged, source_label = row[:4]
            wheel_name, wheel_digest = row[4:] if len(row) > 4 else (None, None)
            digest = _verify(staged, entry.get("sha256"), source_label)
            target = final_directory.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(final_directory.resolve()):
                raise DownloadError(f"Selected output escapes artifact directory: {target}")
            targets.append(target)
            enriched.append({
                "source": source_label,
                "destination": relative.as_posix(),
                "staged": staged,
                "sha256": digest,
                "size_bytes": staged.stat().st_size,
                "wheel": wheel_name,
                "wheel_sha256": wheel_digest,
            })
        manifest_path = final_directory / f".manifest-{artifact_id}.json"
        existing = [path for path in targets if path.exists()]
        previous_manifest = None
        if manifest_path.is_file():
            try:
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise DownloadError(f"Could not read existing artifact manifest {manifest_path}: {error}") from error
            if (
                not isinstance(previous_manifest, dict)
                or previous_manifest.get("schema_version") != 1
                or previous_manifest.get("artifact_id") != artifact_id
                or not isinstance(previous_manifest.get("files"), list)
            ):
                raise DownloadError(
                    f"Existing artifact manifest is invalid or belongs to another artifact: {manifest_path}"
                )
            if any(not isinstance(record, dict) or not isinstance(record.get("path"), str) for record in previous_manifest["files"]):
                raise DownloadError(f"Existing artifact manifest has invalid file records: {manifest_path}")
        if existing and not overwrite:
            raise DownloadError(
                "Refusing to overwrite existing files: "
                + ", ".join(str(path) for path in existing)
                + "; pass --overwrite after review."
            )
        final_directory.mkdir(parents=True, exist_ok=True)
        records = []
        for item, target in zip(enriched, targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_output = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".model-download-", suffix=".tmp", dir=target.parent, delete=False
                ) as output:
                    temp_output = Path(output.name)
                    with item["staged"].open("rb") as staged:
                        shutil.copyfileobj(staged, output, length=1024 * 1024)
                os.replace(temp_output, target)
            finally:
                if temp_output is not None:
                    temp_output.unlink(missing_ok=True)
            record = {
                "source": item["source"],
                "path": (destination / PurePosixPath(item["destination"])).as_posix(),
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            if item["wheel"]:
                record["package_wheel"] = item["wheel"]
                record["package_wheel_sha256"] = item["wheel_sha256"]
            records.append(record)
        download_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_id": artifact_id,
            "provider": source["provider"],
            "source": manifest_source,
            "files": records,
        }
        if previous_manifest is not None:
            previous_files = previous_manifest["files"]
            merged_files = {record["path"]: record for record in previous_files}
            merged_files.update({record["path"]: record for record in download_manifest["files"]})
            manifest = {
                **previous_manifest,
                "updated_at": download_manifest["created_at"],
                "provider": source["provider"],
                "source": manifest_source,
                "files": [merged_files[path] for path in sorted(merged_files)],
            }
        else:
            manifest = download_manifest
        temp_manifest = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".manifest-", suffix=".tmp",
                dir=final_directory, delete=False,
            ) as output:
                temp_manifest = Path(output.name)
                json.dump(manifest, output, indent=2, ensure_ascii=False)
                output.write("\n")
            os.replace(temp_manifest, manifest_path)
        finally:
            if temp_manifest is not None:
                temp_manifest.unlink(missing_ok=True)
    return download_manifest


def _select_artifact_files(
    artifact,
    selected_file_ids=None,
    include_all=False,
):
    source = artifact["source"]
    key = "assets" if source["provider"] == "github_release" else "files"
    entries = source[key]
    if selected_file_ids is not None:
        selected_entries = [entry for entry in entries if entry.get("id") in selected_file_ids]
    elif not include_all and any("default" in entry for entry in entries):
        selected_entries = [
            entry for entry in entries
            if entry.get("default") is True
        ]
    else:
        selected_entries = entries
    if not selected_entries:
        return None
    selected_artifact = dict(artifact)
    selected_source = dict(source)
    selected_source[key] = selected_entries
    selected_artifact["source"] = selected_source
    selected_artifact["_manifest_source"] = source
    return selected_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download pinned Speaker Recognition model artifacts into models/py/ for conversion."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--list", action="store_true", help="List the pinned model files and their provenance")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact id; repeat to select multiple")
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="Exact catalog file id; repeat to select files instead of catalog defaults",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace selected local files after review")
    args = parser.parse_args(argv)
    try:
        project_root = args.project_root.expanduser().resolve()
        if not project_root.is_dir():
            raise DownloadError(f"Project root does not exist: {project_root}")
        catalog, models_root = _load_project(project_root)
        if args.list:
            if args.artifact or args.file is not None or args.overwrite:
                raise DownloadError(
                    "--list cannot be combined with artifact, file, or overwrite options"
                )
            listed_artifacts = catalog["artifacts"]
            result = []
            for artifact in listed_artifacts:
                source = artifact["source"]
                result.append({
                    "id": artifact["id"],
                    "provider": source["provider"],
                    "package": source.get("package"),
                    "version": source.get("version"),
                    "files": [
                        {
                            "id": entry.get("id"),
                            "path": entry.get("path", entry.get("asset")),
                            "default": entry.get("default", False),
                        }
                        for entry in source.get("assets", source.get("files", []))
                    ],
                })
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        available_artifacts = catalog["artifacts"]
        selected = set(args.artifact)
        known = {item["id"] for item in available_artifacts}
        unknown = selected - known
        if unknown:
            raise DownloadError(f"Unknown artifact id(s): {', '.join(sorted(unknown))}")
        artifacts = [item for item in available_artifacts if not selected or item["id"] in selected]
        if not artifacts:
            raise DownloadError("No model artifacts are configured in the project catalog")
        requested_files = set(args.file) if args.file is not None else None
        available_files = {
            entry["id"]
            for artifact in artifacts
            for entry in (
                artifact["source"]["assets"]
                if artifact["source"]["provider"] == "github_release"
                else artifact["source"]["files"]
            )
            if "id" in entry
        }
        unknown_files = sorted((requested_files or set()) - available_files)
        if unknown_files:
            raise DownloadError(f"Unknown or unselected model file id(s): {', '.join(unknown_files)}")
        artifacts = [
            filtered
            for artifact in artifacts
            if (
                filtered := _select_artifact_files(
                    artifact,
                    requested_files,
                    include_all=bool(selected) and requested_files is None,
                )
            ) is not None
        ]
        if not artifacts:
            if requested_files is None:
                raise DownloadError("No default model files are configured; select files with --file.")
            raise DownloadError("No model artifacts contain the selected files")
        for artifact in artifacts:
            manifest = _download_artifact(artifact, models_root, args.overwrite)
            print(f"Downloaded {artifact['id']}: {len(manifest['files'])} file(s)")
            for record in manifest["files"]:
                print(f"  {models_root / record['path']}  sha256={record['sha256']}")
        return 0
    except (DownloadError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
