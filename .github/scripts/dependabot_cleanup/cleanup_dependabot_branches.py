# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Delete stale ``dependabot/*`` branches that no longer have an open pull request.

Dependabot branches are sometimes left behind after their pull request is merged
or closed. This script finds those orphaned branches and removes them.

A ``dependabot/*`` branch is deleted only if it backs no open pull request:

    1. list_dependabot_branches()     -> branches to inspect.
    2. get_recent_open_pr_branches()  -> cheap fast-path subset with a recently
                                         updated open PR.
    3. branch_has_open_pr()           -> authoritative per-branch fallback check,
                                         so an open PR of any age is never missed.
    4. delete_branch()                -> the side effect.
    5. run_cleanup()                  -> wires the steps together and reports.

Run it manually (dry-run is the default, so nothing is deleted):

    GITHUB_TOKEN=<token> GITHUB_REPOSITORY=openvinotoolkit/openvino \\
        python3 cleanup_dependabot_branches.py --dry-run

Pass ``--no-dry-run`` (or set ``DRY_RUN=false``) to actually delete branches.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from github import Auth, Github
from github.GithubException import GithubException

DEPENDABOT_BRANCH_PREFIX = "dependabot/"

# Only open PRs updated within this many days are treated as "still in use".
# The repository can have hundreds of open PRs, so this bounds how many we read.
DEFAULT_OPEN_PR_MAX_AGE_DAYS = 30

# Exit codes, kept explicit so the CI step is easy to reason about.
EXIT_OK = 0
EXIT_DELETION_FAILED = 1
EXIT_BAD_CONFIG = 2

logger = logging.getLogger("dependabot_cleanup")


@dataclass
class Config:
    """Everything the script needs to run, gathered in one place."""

    repository_name: str
    token: str
    dry_run: bool
    open_pr_max_age_days: int


@dataclass
class CleanupResult:
    """A simple record of what happened, used for the final summary."""

    scanned: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core rules and GitHub queries
# --------------------------------------------------------------------------- #
def is_dependabot_branch(branch_name: str) -> bool:
    """Return True for branches managed by Dependabot."""
    return branch_name.startswith(DEPENDABOT_BRANCH_PREFIX)


def get_recent_open_pr_branches(repository, max_age_days: int) -> set[str]:
    """Return head branch names of open PRs updated within ``max_age_days``.

    With hundreds of open PRs in the repository, we sort by
    most-recently-updated and stop at the first PR older than the cutoff instead
    of paging through all of them. This is a subset of all open-PR branches, so
    it must never be the sole basis for deletion -- see :func:`branch_has_open_pr`.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    open_pr_branches: set[str] = set()
    for pull in repository.get_pulls(
        state="open", sort="updated", direction="desc"
    ):
        updated_at = pull.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        if updated_at < cutoff:
            break
        open_pr_branches.add(pull.head.ref)
    return open_pr_branches


def branch_has_open_pr(repository, owner_login: str, branch_name: str) -> bool:
    """Return True if ``branch_name`` currently backs at least one open PR.

    This is the authoritative safety check, independent of the recent-PR window.
    It filters server-side by head branch (``owner:branch``) so it inspects only
    the PR(s) for this one branch rather than every open PR in the repository.
    """
    open_pulls = repository.get_pulls(
        state="open", head=f"{owner_login}:{branch_name}"
    )
    return open_pulls.totalCount > 0


def list_dependabot_branches(repository) -> list[str]:
    """Return the names of all ``dependabot/*`` branches in the repository."""
    return [
        branch.name
        for branch in repository.get_branches()
        if is_dependabot_branch(branch.name)
    ]


def delete_branch(repository, branch_name: str) -> bool:
    """Delete a branch via the git refs API. Return True on success."""
    try:
        repository.get_git_ref(f"heads/{branch_name}").delete()
        return True
    except GithubException as error:
        logger.error(f'Failed to delete branch "{branch_name}": {error}')
        return False


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_cleanup(repository, dry_run: bool, open_pr_max_age_days: int) -> CleanupResult:
    """Scan dependabot branches and delete the stale ones.

    When ``dry_run`` is True the script only logs what it *would* delete.
    """
    recent_open_pr_branches = get_recent_open_pr_branches(
        repository, open_pr_max_age_days
    )
    logger.info(
        f"Found {len(recent_open_pr_branches)} open pull request branch(es) "
        f"updated within {open_pr_max_age_days} day(s)"
    )

    owner_login = repository.full_name.split("/")[0]
    dependabot_branches = list_dependabot_branches(repository)
    logger.info(
        f'Found {len(dependabot_branches)} "{DEPENDABOT_BRANCH_PREFIX}" '
        f"branch(es) to inspect"
    )

    result = CleanupResult(scanned=dependabot_branches)

    for branch_name in dependabot_branches:
        # Fast path: branch is known to back a recently updated open PR.
        if branch_name in recent_open_pr_branches:
            logger.info(f'Keep    "{branch_name}" (still has an open PR)')
            result.kept.append(branch_name)
            continue

        # Authoritative check: never delete a branch that backs an open PR,
        # even one older than the recent-scan window.
        if branch_has_open_pr(repository, owner_login, branch_name):
            logger.info(f'Keep    "{branch_name}" (still has an open PR)')
            result.kept.append(branch_name)
            continue

        if dry_run:
            logger.info(f'Dry-run "{branch_name}" (would be deleted)')
            result.deleted.append(branch_name)
            continue

        logger.info(f'Delete  "{branch_name}" (no open PR)')
        if delete_branch(repository, branch_name):
            result.deleted.append(branch_name)
        else:
            result.failed.append(branch_name)

    return result


def log_summary(result: CleanupResult, dry_run: bool) -> None:
    """Print a compact, human-friendly summary of the run."""
    deleted_label = "Would delete" if dry_run else "Deleted"

    logger.info("-" * 60)
    logger.info("Summary")
    logger.info(f"  Scanned            : {len(result.scanned)}")
    logger.info(f"  Kept (open PR)     : {len(result.kept)}")
    logger.info(f"  {deleted_label:<18} : {len(result.deleted)}")
    for branch_name in result.deleted:
        logger.info(f"      - {branch_name}")
    if result.failed:
        logger.info(f"  Failed to delete   : {len(result.failed)}")
        for branch_name in result.failed:
            logger.info(f"      - {branch_name}")
    logger.info("-" * 60)


# --------------------------------------------------------------------------- #
# Configuration and entry point
# --------------------------------------------------------------------------- #
def read_bool_env(name: str) -> bool | None:
    """Read a boolean-ish environment variable.

    Returns None if it is unset/empty, or if its value is not a recognized
    boolean spelling -- a typo (e.g. ``DRY_RUN=flase``) must not silently be
    treated as False, since that would make ``dry_run`` fall back to its safe
    default of True rather than accidentally disabling it.
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    logger.warning(f'Ignoring unrecognized boolean {name}="{value}"')
    return None


def read_int_env(name: str) -> int | None:
    """Read an integer environment variable, or None if unset/empty/invalid."""
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value.strip())
    except ValueError:
        logger.warning(f'Ignoring non-integer {name}="{value}"')
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The dry-run default is resolved from the ``DRY_RUN`` environment variable and
    falls back to True (the safe choice) when it is not set. An explicit
    ``--dry-run`` / ``--no-dry-run`` flag always wins over the environment.
    """
    dry_run_default = read_bool_env("DRY_RUN")
    if dry_run_default is None:
        dry_run_default = True

    open_pr_max_age_default = read_int_env("OPEN_PR_MAX_AGE_DAYS")
    if open_pr_max_age_default is None:
        open_pr_max_age_default = DEFAULT_OPEN_PR_MAX_AGE_DAYS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--repository-name",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Repository in OWNER/REPO format "
             "(defaults to the GITHUB_REPOSITORY env var)",
    )
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only log which branches would be deleted (default)",
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually delete the stale branches",
    )
    parser.add_argument(
        "--open-pr-max-age-days",
        type=int,
        default=open_pr_max_age_default,
        help="Only treat open PRs updated within this many days as keeping "
             "their branch alive (defaults to the OPEN_PR_MAX_AGE_DAYS env var "
             f"or {DEFAULT_OPEN_PR_MAX_AGE_DAYS})",
    )
    parser.set_defaults(dry_run=dry_run_default)
    return parser.parse_args()


def build_config() -> Config | None:
    """Build the Config from CLI args and environment, or None if it is invalid."""
    args = parse_args()

    if not args.repository_name:
        logger.error(
            "Repository name is required "
            "(pass --repository-name or set GITHUB_REPOSITORY)"
        )
        return None

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("The GITHUB_TOKEN environment variable is required")
        return None

    return Config(
        repository_name=args.repository_name,
        token=token,
        dry_run=args.dry_run,
        open_pr_max_age_days=args.open_pr_max_age_days,
    )


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
    )

    config = build_config()
    if config is None:
        return EXIT_BAD_CONFIG

    logger.info(
        f'Cleaning dependabot branches in "{config.repository_name}" '
        f"(dry_run={config.dry_run})"
    )

    github = Github(auth=Auth.Token(token=config.token))
    try:
        repository = github.get_repo(config.repository_name)
        result = run_cleanup(
            repository, config.dry_run, config.open_pr_max_age_days
        )
    finally:
        github.close()

    log_summary(result, config.dry_run)

    # Only fail the run if a real deletion attempt failed.
    return EXIT_DELETION_FAILED if result.failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
