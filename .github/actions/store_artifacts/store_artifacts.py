# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import git
import shutil
from contextlib import contextmanager
from pathlib import Path
from datetime import timezone
from manifest_manager import Manifest, Repository, Component


def parse_args():
    parser = argparse.ArgumentParser(description='Returns product components changed in a given PR or commit')
    parser.add_argument('-e', '--event_name', help='Name of GitHub event', required=False)
    parser.add_argument('-b', '--branch_name', help='Name of GitHub branch', required=False)
    parser.add_argument('-s', '--commit_sha', help='Commit hash for which artifacts were generated', required=False)
    parser.add_argument('-a', '--artifacts', type=str, help='Paths to artifacts to store (files/dirs)', required=True)
    parser.add_argument('-r', '--repos', type=str, help='Paths to repositories used to generate artifacts',
                        required=True)
    parser.add_argument('--storage_dir', help='Directory name to store artifacts in', required=True)
    parser.add_argument('--storage_root', help='Root path of the storage to place artifacts to', required=True)
    parser.add_argument('--target_arch', help='Architecture for which artifacts were generated', required=True)
    parser.add_argument('--build_type', help='Build type: release | debug | release_with_debug', required=True)
    args = parser.parse_args()
    return args


def init_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


def set_github_output(name: str, value: str, github_output_var_name: str = 'GITHUB_OUTPUT'):
    """Sets output variable for a GitHub Action"""
    logger = logging.getLogger(__name__)
    # In an environment variable "GITHUB_OUTPUT" GHA stores path to a file to write outputs to
    with open(os.environ.get(github_output_var_name), 'a+') as file:
        logger.info(f"Add {name}={value} to {github_output_var_name}")
        print(f'{name}={value}', file=file)


@contextmanager
def preserve_stats_context():
    """
    Workaround for copying to samba share on Linux
    to avoid issues while setting Linux permissions.
    """
    _orig_copystat = shutil.copystat
    shutil.copystat = lambda x, y, follow_symlinks=True: x
    try:
        yield
    finally:
        shutil.copystat = _orig_copystat


def rotate_dir(directory: Path) -> bool:
    """
    Renames directory if exists:
    dir -> dir_1
    """
    log = logging.getLogger('rotate_dir')

    if not directory.exists():
        return False

    dir_parent = directory.parent
    dir_name = directory.name
    max_dir_num = 0
    for redir in dir_parent.iterdir():
        dir_num = redir.name.split('_')[-1]
        if redir.name.startswith(dir_name) and dir_num.isdigit() and int(dir_num) > max_dir_num:
            max_dir_num = int(dir_num)

    duplicate = dir_parent / f'{dir_name}_{max_dir_num + 1}'
    log.info(f"Move previous directory to {duplicate}")
    directory.rename(duplicate)
    return True


def get_repo_data(repo_dir: str | Path) -> dict:
    repo = git.Repo(str(repo_dir))
    repo_url = next(repo.remote().urls)
    repo_name_match = re.search(r'github\.com/[^/]+/([^/]+)', repo_url)
    repo_name = repo_name_match.group(1) if repo_name_match else None

    trigger_repo_url = f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
    is_trigger_repo = repo_url == trigger_repo_url

    branch = os.getenv('GITHUB_REF_NAME') if is_trigger_repo else repo.references[0].name
    target_brach = os.getenv('BASE_BRANCH') if is_trigger_repo else None
    revision = os.getenv('PR_HEAD_SHA') or os.getenv('GITHUB_SHA') if is_trigger_repo else repo.head.commit.hexsha
    target_revision = os.getenv('BASE_SHA') if is_trigger_repo else None
    # Commit time of a merge commit (in case of PR merged to target)
    # TODO: Save commit time of a head commit in PR as well?
    commit_time = repo.head.commit.committed_datetime.astimezone(timezone.utc)
    merge_target = branch.endswith('/merge')
    return {
        'name': repo_name,
        'url': repo_url,
        'branch': branch,
        'target_branch': target_brach,
        'revision': revision,
        'target_revision': target_revision,
        'commit_time': commit_time,
        'merge_target': merge_target,
        'trigger': is_trigger_repo,
    }


def generate_manifest(version: str, repos: list, product_type: str, event_type: str, build_type: str,
                      target_arch: str) -> Manifest:
    manifest = Manifest()
    version = version
    component_name = 'dldt' # historical, keep for internal compatibility
    repositories = []
    for repo_dir in repos:
        repo_data = get_repo_data(repo_dir)
        repositories.append(Repository(**repo_data))
    # TODO: add wheels product version to custom params
    component = Component(name=component_name, version=version, product_type=product_type, target_arch=target_arch,
                          build_type=build_type, build_event=event_type, repositories=repositories)

    manifest.add_component(component)
    return manifest


def main():
    init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()

    event_name = args.event_name or os.getenv('GITHUB_EVENT_NAME')
    branch_name = args.branch_name or os.getenv('GITHUB_BASE_REF') or os.getenv('GITHUB_REF_NAME')
    merge_queue_matcher = re.search(r'gh-readonly-queue/(.*?)/pr-', branch_name)
    if merge_queue_matcher:
        branch_name = merge_queue_matcher.group(1)

    commit_hash = args.commit_sha or os.getenv('GITHUB_EVENT_PULL_REQUEST_HEAD_SHA') or os.getenv('GITHUB_SHA')
    event_type = 'pre_commit' if event_name == 'pull_request' else 'commit'
    storage_root = args.storage_root or os.getenv('ARTIFACTS_SHARE')

    storage = Path(storage_root) / 'dldt' / branch_name / event_type / commit_hash / args.storage_dir
    set_github_output("artifacts_storage_path", str(storage))

    logger.info(f"Storing artifacts to {storage}")
    rotate_dir(storage)  # TODO: use more stable approach to handle storing artifacts from re-runs

    error_found = False
    for artifact in args.artifacts.split():
        artifact_path = Path(artifact)
        logger.debug(f"Copying {artifact_path} to {storage / artifact_path.name}")
        try:
            with preserve_stats_context():
                if artifact_path.is_dir():
                    shutil.copytree(artifact_path, storage / artifact_path.name)
                else:
                    storage.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(artifact_path, storage / artifact_path.name)
        except Exception as e:
            logger.error(f'Failed to copy {artifact}: {e}')
            error_found = True

    # TODO: move this and manifest generation before copying artifacts?
    github_server = os.getenv('GITHUB_SERVER_URL')
    if github_server:  # If running from GHA context
        # TODO: write an exact job link, but it's not trivial to get
        workflow_link = f"{github_server}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
        with open(storage / 'workflow_link.txt', 'w') as file:
            file.write(workflow_link)

    # Generate manifest
    version = 'TBD' # TODO: generate version
    repos = args.repos.split()
    manifest = generate_manifest(version, repos, args.storage_dir, event_type, args.build_type, args.target_arch)
    manifest.save_manifest('manifest.yml') # Locally, to upload to GitHub artifacts
    manifest.save_manifest(storage / 'manifest.yml') # Remotely

    logger.debug(f"Copying finished")
    (storage / 'copying_finished').touch()
    if error_found:
        sys.exit(1)


if __name__ == '__main__':
    main()
