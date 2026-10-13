#!/usr/bin/env python

"""
Snapshot, record and freeze Tukey runs so every result has immutable provenance.

Every job runs from `runs/<run_id>/src/`, a `git archive` copy of the project
directory at a committed SHA, so edits to the working tree during a run cannot
leak into it. `provenance.json` records what produced the run and
`SHA256SUMS` fixes its outputs; the directory is made read-only at the end.

Confirmatory runs (the single evaluation on a sealed test set) additionally
require a clean tree whose HEAD and Docker image match a freeze record.

The script runs on the host (stdlib + helpers only) and is called by
`run_job.sh`.

# Usage Example

- Snapshot the current commit for a new run:
> tukey_runs.py prepare --run_id m3_seed0

- Close a run after its command finished with exit code 0:
> tukey_runs.py finalize --run_id m3_seed0 --exit_code 0

- Re-check the hashes of a finished run:
> tukey_runs.py verify --run_id m3_seed0

- Freeze the current commit for confirmatory testing:
> tukey_runs.py freeze --freeze_name m3 --files configs/m3.json
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import stat
import subprocess
from typing import Any, Dict, List

import helpers.hdbg as hdbg
import helpers.hparser as hparser

_LOG = logging.getLogger(__name__)

# Docker image every job runs in.
_IMAGE_NAME = "gpsaggese/umd_project_tukey"


# #############################################################################
# Helpers
# #############################################################################


def _run(cmd: List[str], *, cwd: str) -> str:
    """
    Run a command and return its stripped standard output.

    :param cmd: command and arguments
    :param cwd: working directory
    :return: standard output
    """
    output = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout
    return output.strip()


def _sha256_file(path: str) -> str:
    """
    Compute the SHA256 digest of a file.

    :param path: file to hash
    :return: hex digest
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_files(root: str) -> List[str]:
    """
    List every file under a directory, as sorted paths relative to it.

    :param root: directory to walk
    :return: sorted relative file paths
    """
    paths = []
    for dir_path, _, file_names in os.walk(root):
        for file_name in file_names:
            full_path = os.path.join(dir_path, file_name)
            paths.append(os.path.relpath(full_path, root))
    return sorted(paths)


def _hash_tree(root: str) -> str:
    """
    Hash a directory's content (file paths and bytes) into one digest.

    :param root: directory to hash
    :return: hex digest that changes if any file or name changes
    """
    digest = hashlib.sha256()
    for rel_path in _list_files(root):
        digest.update(rel_path.encode())
        digest.update(_sha256_file(os.path.join(root, rel_path)).encode())
    return digest.hexdigest()


def _get_image_id() -> str:
    """
    Get the ID of the locally built Tukey image.

    :return: image ID (e.g., "sha256:3a4c...")
    """
    image_id = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", _IMAGE_NAME],
        cwd=".",
    )
    return image_id


def _get_git_state(project_dir: str) -> Dict[str, Any]:
    """
    Describe the commit and cleanliness of the project directory.

    :param project_dir: project directory inside the git repo
    :return: commit SHA, helpers submodule SHA and dirty flag
    """
    commit = _run(["git", "rev-parse", "HEAD"], cwd=project_dir)
    helpers_sha = _run(
        ["git", "rev-parse", "HEAD:helpers_root"], cwd=project_dir
    )
    # A non-empty porcelain status under the project dir means uncommitted work.
    status = _run(["git", "status", "--porcelain", "--", "."], cwd=project_dir)
    git_state = {
        "commit": commit,
        "helpers_root_commit": helpers_sha,
        "dirty": bool(status),
    }
    return git_state


def _now_utc() -> str:
    """
    Return the current UTC time in ISO format.

    :return: timestamp (e.g., "2026-10-13T05:46:15+00:00")
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )
    return now


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    """
    Write a JSON object with sorted keys.

    :param path: destination file
    :param obj: JSON-serializable dictionary
    """
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


# #############################################################################
# Freeze records
# #############################################################################


def _freeze_path(project_dir: str, freeze_name: str) -> str:
    """
    Return the path of a freeze record.

    :param project_dir: project directory
    :param freeze_name: freeze name (e.g., "m3")
    :return: path of `results/<freeze_name>/FREEZE.json`
    """
    path = os.path.join(project_dir, "results", freeze_name, "FREEZE.json")
    return path


def _freeze(project_dir: str, freeze_name: str, files: List[str]) -> None:
    """
    Record the commit, image and file hashes that a confirmatory run must match.

    :param project_dir: project directory
    :param freeze_name: freeze name (e.g., "m3")
    :param files: files to pin (configs, checkpoints, sealed-set manifests),
        relative to the project directory
    """
    git_state = _get_git_state(project_dir)
    hdbg.dassert(
        not git_state["dirty"],
        "Commit all changes before freezing '%s'",
        freeze_name,
    )
    record = {
        "freeze_name": freeze_name,
        "created_utc": _now_utc(),
        "commit": git_state["commit"],
        "helpers_root_commit": git_state["helpers_root_commit"],
        "image_id": _get_image_id(),
        "files": {
            rel_path: _sha256_file(os.path.join(project_dir, rel_path))
            for rel_path in sorted(files)
        },
    }
    path = _freeze_path(project_dir, freeze_name)
    hdbg.dassert(
        not os.path.exists(path), "Freeze record '%s' already exists", path
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, record)
    _LOG.info("Wrote freeze record '%s' at commit '%s'", path, record["commit"])


def _check_confirmatory(project_dir: str, freeze_name: str) -> None:
    """
    Check that the current state matches a freeze record exactly.

    :param project_dir: project directory
    :param freeze_name: freeze name to check against
    """
    path = _freeze_path(project_dir, freeze_name)
    hdbg.dassert_file_exists(path)
    with open(path) as fh:
        record = json.load(fh)
    git_state = _get_git_state(project_dir)
    hdbg.dassert(not git_state["dirty"], "Confirmatory runs need a clean tree")
    hdbg.dassert_eq(git_state["commit"], record["commit"])
    hdbg.dassert_eq(_get_image_id(), record["image_id"])
    for rel_path, digest in record["files"].items():
        actual = _sha256_file(os.path.join(project_dir, rel_path))
        hdbg.dassert_eq(
            actual, digest, "File '%s' changed since freeze", rel_path
        )


# #############################################################################
# Run lifecycle
# #############################################################################


def _prepare(
    project_dir: str, run_id: str, confirmatory: str, command: str
) -> None:
    """
    Create `runs/<run_id>/` with a source snapshot and the initial provenance.

    :param project_dir: project directory
    :param run_id: unique run name
    :param confirmatory: freeze name the run must match; "" for exploratory runs
    :param command: the command the job runs inside the snapshot
    """
    run_dir = os.path.join(project_dir, "runs", run_id)
    hdbg.dassert(not os.path.exists(run_dir), "Run dir '%s' exists", run_dir)
    if confirmatory:
        _check_confirmatory(project_dir, confirmatory)
    git_state = _get_git_state(project_dir)
    # Export the committed project directory (not the working tree). The
    # `<commit>:<dir>` tree-ish archives that subtree with paths relative to
    # it. `git archive` must run from the repo root: from a subdirectory it
    # also limits the output to that subdirectory, which yields an empty tar.
    rel_dir = _run(["git", "rev-parse", "--show-prefix"], cwd=project_dir)
    repo_root = _run(["git", "rev-parse", "--show-toplevel"], cwd=project_dir)
    tree_ish = f"{git_state['commit']}:{rel_dir.rstrip('/')}"
    archive = subprocess.run(
        ["git", "archive", "--format=tar", tree_ish],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    # An empty tar is 10240 bytes of zeros: refuse to start from no source.
    hdbg.dassert_lt(
        10240, len(archive), "Empty source archive for '%s'", tree_ish
    )
    # Create the run dir only once the archive exists, so a failure leaves
    # nothing behind.
    src_dir = os.path.join(run_dir, "src")
    os.makedirs(src_dir)
    subprocess.run(["tar", "-x", "-C", src_dir], input=archive, check=True)
    provenance = {
        "run_id": run_id,
        "command": command,
        "created_utc": _now_utc(),
        "confirmatory": confirmatory,
        "commit": git_state["commit"],
        "helpers_root_commit": git_state["helpers_root_commit"],
        "dirty_working_tree": git_state["dirty"],
        "source_sha256": _hash_tree(src_dir),
        "image_id": _get_image_id(),
        "host_cpu_count": os.cpu_count(),
    }
    _write_json(os.path.join(run_dir, "provenance.json"), provenance)
    _LOG.info("Prepared run '%s' from commit '%s'", run_id, provenance["commit"])


def _finalize(project_dir: str, run_id: str, exit_code: int) -> None:
    """
    Record the end of a run, hash its outputs and make it read-only.

    :param project_dir: project directory
    :param run_id: run name
    :param exit_code: exit code of the job's command
    """
    run_dir = os.path.join(project_dir, "runs", run_id)
    path = os.path.join(run_dir, "provenance.json")
    with open(path) as fh:
        provenance = json.load(fh)
    provenance["finished_utc"] = _now_utc()
    provenance["exit_code"] = exit_code
    _write_json(path, provenance)
    # Hash every file so later tampering or truncation is detectable.
    lines = [
        f"{_sha256_file(os.path.join(run_dir, rel_path))}  {rel_path}"
        for rel_path in _list_files(run_dir)
        if rel_path != "SHA256SUMS"
    ]
    with open(os.path.join(run_dir, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # Remove write permission from every file and directory in the run.
    for dir_path, _, file_names in os.walk(run_dir):
        for name in file_names + [""]:
            target = os.path.join(dir_path, name)
            mode = os.stat(target).st_mode
            os.chmod(
                target, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
    _LOG.info("Finalized run '%s' with exit code %s", run_id, exit_code)


def _verify(project_dir: str, run_id: str) -> None:
    """
    Check every file of a finished run against its `SHA256SUMS`.

    :param project_dir: project directory
    :param run_id: run name
    """
    run_dir = os.path.join(project_dir, "runs", run_id)
    with open(os.path.join(run_dir, "SHA256SUMS")) as fh:
        lines = [line for line in fh.read().splitlines() if line]
    for line in lines:
        digest, rel_path = line.split("  ", 1)
        actual = _sha256_file(os.path.join(run_dir, rel_path))
        hdbg.dassert_eq(actual, digest, "File '%s' changed", rel_path)
    _LOG.info("Run '%s' verified: %s files unchanged", run_id, len(lines))


# #############################################################################


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=hparser.CustomHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "finalize",
            "verify",
            "freeze",
            "check_confirmatory",
        ],
        help="Step of the run lifecycle to execute",
    )
    parser.add_argument("--run_id", action="store", default="", help="Run name")
    parser.add_argument(
        "--exit_code", action="store", type=int, default=0, help="Job exit code"
    )
    parser.add_argument(
        "--confirmatory",
        action="store",
        default="",
        help="Freeze name a confirmatory run must match",
    )
    parser.add_argument(
        "--freeze_name", action="store", default="", help="Freeze record name"
    )
    parser.add_argument(
        "--command",
        action="store",
        default="",
        help="Command the job runs (recorded in the provenance)",
    )
    parser.add_argument(
        "--files", action="store", nargs="*", default=[], help="Files to freeze"
    )
    parser.add_argument(
        "--project_dir",
        action="store",
        default=".",
        help="Project directory (run from it by default)",
    )
    hparser.add_verbosity_arg(parser)
    return parser


def _main(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args()
    hdbg.init_logger(verbosity=args.log_level, use_exec_path=True)
    project_dir = os.path.abspath(args.project_dir)
    if args.action in ("prepare", "finalize", "verify"):
        hdbg.dassert_ne(args.run_id, "", "--run_id is required")
    if args.action == "prepare":
        _prepare(project_dir, args.run_id, args.confirmatory, args.command)
    elif args.action == "finalize":
        _finalize(project_dir, args.run_id, args.exit_code)
    elif args.action == "verify":
        _verify(project_dir, args.run_id)
    elif args.action == "freeze":
        hdbg.dassert_ne(args.freeze_name, "", "--freeze_name is required")
        _freeze(project_dir, args.freeze_name, args.files)
    else:
        hdbg.dassert_ne(args.freeze_name, "", "--freeze_name is required")
        _check_confirmatory(project_dir, args.freeze_name)


if __name__ == "__main__":
    _main(_parse())
