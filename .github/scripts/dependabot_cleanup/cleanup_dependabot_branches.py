# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Delete stale ``dependabot/*`` branches that no longer have an open pull request.

Dependabot branches are sometimes left behind after their pull request is merged
or closed. This script finds those orphaned branches and removes them.

The logic is intentionally small and split into clear steps so it is easy to read
and to extend:

    1. list_dependabot_branches()  -> which branches do we care about?
    2. get_open_pr_branches()      -> which branches are still in use?
    3. is_branch_stale()           -> the single rule that decides deletion.
    4. delete_branch()             -> the side effect.
    5. run_cleanup()               -> wires the steps together and reports.

To change *what counts as stale* (for example, also require the branch to be a
few days old), you only need to change ``is_branch_stale``.

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


def is_branch_stale(branch_name: str, open_pr_branches: set[str]) -> bool:
    """Decide whether a dependabot branch is safe to delete.

    A branch is stale when it does NOT back an open pull request. That covers
    branches whose PR was merged or closed, as well as branches that never had a
    PR. Branches with an open PR are always kept.

    This is the single place to change the deletion policy. For example, to also
    keep very recent branches you would add that extra condition here.
    """
    return branch_name not in open_pr_branches


def get_open_pr_branches(repository, max_age_days: int) -> set[str]:
    """Return head branch names of open PRs updated within ``max_age_days``.

    The repository routinely has hundreds of open PRs, so instead of paging
    through all of them we ask GitHub to sort by most-recently-updated and stop
    as soon as we reach a PR older than the cutoff. Dependabot keeps active PRs
    fresh (it rebases them as the base branch moves), so a PR that has not been
    touched within this window is treated as no longer keeping its branch alive.
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
    open_pr_branches = get_open_pr_branches(repository, open_pr_max_age_days)
    logger.info(
        f"Found {len(open_pr_branches)} open pull request branch(es) "
        f"updated within {open_pr_max_age_days} day(s)"
    )

    dependabot_branches = list_dependabot_branches(repository)
    logger.info(
        f'Found {len(dependabot_branches)} "{DEPENDABOT_BRANCH_PREFIX}" '
        f"branch(es) to inspect"
    )

    result = CleanupResult(scanned=dependabot_branches)

    for branch_name in dependabot_branches:
        if not is_branch_stale(branch_name, open_pr_branches):
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
    """Read a boolean-ish environment variable, or None if it is unset/empty."""
    value = os.environ.get(name)
    if not value:
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


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
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only log which branches would be deleted (default)",
    )
    parser.add_argument(
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
