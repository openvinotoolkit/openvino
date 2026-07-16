# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Delete stale ``dependabot/*`` branches that no longer have an open PR.

Sometimes Dependabot branches are not removed automatically after their PR is
merged or closed. This script enumerates every branch whose name starts with
``dependabot/`` and deletes the ones that are safe to remove.

A branch is considered *stale* (safe to delete) when there is NO open pull
request using it as the head branch. Branches that still have an open PR are
always kept.

Run it manually (dry-run is the default here to be safe):

    GITHUB_TOKEN=<token> GITHUB_REPOSITORY=openvinotoolkit/openvino \\
        python3 cleanup_dependabot_branches.py --dry-run

To actually delete branches pass ``--no-dry-run`` (or set ``DRY_RUN=false``).
"""

import argparse
import logging
import os
import sys

from github import Github, Auth
from github.GithubException import GithubException

BRANCH_PREFIX = 'dependabot/'

LOGGER = logging.getLogger('dependabot_cleanup')


def init_logger() -> None:
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
        format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s',
        datefmt='%m-%d-%Y %H:%M:%S',
    )


def _env_flag(name: str) -> bool | None:
    """Parse a boolean-ish environment variable. Returns None if unset."""
    value = os.environ.get(name)
    if value is None or value == '':
        return None
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-r',
        '--repository-name',
        type=str,
        default=os.environ.get('GITHUB_REPOSITORY'),
        help='Repository name in the OWNER/REPOSITORY format '
             '(defaults to the GITHUB_REPOSITORY env var)',
    )

    # --dry-run / --no-dry-run. The default is resolved from the DRY_RUN env
    # var when neither flag is passed, otherwise it falls back to True (safe).
    dry_run_default = _env_flag('DRY_RUN')
    if dry_run_default is None:
        dry_run_default = True

    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='Only log which branches would be deleted, do not delete anything',
    )
    parser.add_argument(
        '--no-dry-run',
        dest='dry_run',
        action='store_false',
        help='Actually delete the stale branches',
    )
    parser.set_defaults(dry_run=dry_run_default)
    return parser.parse_args()


def get_open_pr_head_refs(gh_repo) -> set[str]:
    """Return the set of head branch names that currently have an open PR.

    Pagination is handled transparently by PyGithub's ``PaginatedList``.
    """
    open_heads: set[str] = set()
    for pull in gh_repo.get_pulls(state='open'):
        # ``head.ref`` is the branch name within the head repository.
        open_heads.add(pull.head.ref)
    return open_heads


def get_dependabot_branches(gh_repo) -> list[str]:
    """Return all branch names starting with the dependabot prefix."""
    branches: list[str] = []
    for branch in gh_repo.get_branches():
        if branch.name.startswith(BRANCH_PREFIX):
            branches.append(branch.name)
    return branches


def delete_branch(gh_repo, branch_name: str) -> bool:
    """Delete a branch via the git refs API. Returns True on success."""
    try:
        ref = gh_repo.get_git_ref(f'heads/{branch_name}')
        ref.delete()
        return True
    except GithubException as exc:
        LOGGER.error('FAILED TO DELETE BRANCH "%s": %s', branch_name, exc)
        return False


def main() -> int:
    init_logger()
    args = get_arguments()

    if not args.repository_name:
        LOGGER.error(
            'Repository name is required (pass --repository-name or set '
            'GITHUB_REPOSITORY)'
        )
        return 2

    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        LOGGER.error('GITHUB_TOKEN env var is required')
        return 2

    LOGGER.info(
        'Starting dependabot branch cleanup for "%s" (dry_run=%s)',
        args.repository_name,
        args.dry_run,
    )

    github = Github(auth=Auth.Token(token=token))
    try:
        gh_repo = github.get_repo(full_name_or_id=args.repository_name)

        open_heads = get_open_pr_head_refs(gh_repo)
        LOGGER.info('Found %d open PR head branches', len(open_heads))

        dependabot_branches = get_dependabot_branches(gh_repo)
        LOGGER.info(
            'Found %d "%s" branches to inspect',
            len(dependabot_branches),
            BRANCH_PREFIX,
        )

        kept: list[str] = []
        deleted: list[str] = []
        failed: list[str] = []

        for branch_name in dependabot_branches:
            if branch_name in open_heads:
                LOGGER.info('KEEP   "%s" (has an open PR)', branch_name)
                kept.append(branch_name)
                continue

            if args.dry_run:
                LOGGER.info(
                    'DRY-RUN would delete "%s" (no open PR)', branch_name
                )
                deleted.append(branch_name)
                continue

            LOGGER.info('DELETE "%s" (no open PR)', branch_name)
            if delete_branch(gh_repo, branch_name):
                deleted.append(branch_name)
            else:
                failed.append(branch_name)
    finally:
        github.close()

    LOGGER.info('=' * 60)
    LOGGER.info('SUMMARY')
    LOGGER.info('  Branches scanned: %d', len(dependabot_branches))
    LOGGER.info('  Branches kept (open PR): %d', len(kept))
    LOGGER.info(
        '  Branches %s: %d',
        'that would be deleted' if args.dry_run else 'deleted',
        len(deleted),
    )
    if deleted:
        for branch_name in deleted:
            LOGGER.info('    - %s', branch_name)
    LOGGER.info('  Branch deletions failed: %d', len(failed))
    LOGGER.info('=' * 60)

    # Only fail the run if a real deletion attempt failed.
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
