# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import shutil
from contextlib import contextmanager
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from common import artifact_utils, action_utils
from common.constants import PlatformMapping, PlatformKey


def parse_args():
    parser = argparse.ArgumentParser(description='Stores given artifacts on a shared drive')
    parser.add_argument('-a', '--artifacts', type=str, help='Paths to artifacts to store (files/dirs)', required=True)
    artifact_utils.add_common_args(parser)
    args = parser.parse_args()
    return args


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


def generate_sha256sum(file_path: str | Path) -> str:
    """
    Generates the SHA-256 checksum for the given file.

    :param file_path: Path to the file
    :return: SHA-256 checksum as a hexadecimal string
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def store_checksums(artifact_path: Path) -> None:
    """
    Generates SHA-256 checksums for a given artifact and stores them in separate files.

    :param artifact_path: Path to either a single artifact file or the directory with files to generate checksums for
    """
    files = [path for path in artifact_path.rglob('*') if path.is_file] if artifact_path.is_dir() else [artifact_path]
    for file in files:
        hashsum_filepath = file.with_suffix('.sha256')
        with open(hashsum_filepath, 'w') as hashsum_file:
            hashsum_file.write(generate_sha256sum(file))


def main():
    action_utils.init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()

    storage_root = args.storage_root or os.getenv('ARTIFACTS_SHARE')
    storage_dir = args.storage_dir or PlatformMapping[PlatformKey[args.platform.upper()]].value
    storage = artifact_utils.get_storage_dir(storage_dir, args.commit_sha, args.storage_root, args.branch_name,
                                             args.event_name, args.product_name)
    action_utils.set_github_output("artifacts_storage_path", str(storage))

    logger.info(f"Storing artifacts to {storage}")
    rotate_dir(storage)  # TODO: use more stable approach to handle storing artifacts from re-runs

    error_found = False
    for artifact in args.artifacts.split():
        artifact_path = Path(artifact)

        logger.debug(f"Calculating checksum for {artifact_path}")
        try:
            store_checksums(artifact_path)
        except Exception as e:
            logger.error(f'Failed to calculate checksum for {artifact}: {e}')
            error_found = True

        logger.debug(f"Copying {artifact_path} to {storage / artifact_path.name}")
        try:
            with preserve_stats_context():
                if artifact_path.is_dir():
                    shutil.copytree(artifact_path, storage / artifact_path.name)
                else:
                    storage.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(artifact_path, storage / artifact_path.name)
                    if artifact_path.with_suffix('.sha256').exists():
                        shutil.copy2(artifact_path.with_suffix('.sha256'),
                                     storage / artifact_path.with_suffix('.sha256').name)
        except Exception as e:
            logger.error(f'Failed to copy {artifact}: {e}')
            error_found = True

    github_server = os.getenv('GITHUB_SERVER_URL')
    if github_server:  # If running from GHA context
        # TODO: write an exact job link, but it's not trivial to get
        workflow_link = f"{github_server}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
        workflow_link_file = storage / 'workflow_link.txt'
        with open(workflow_link_file, 'w') as file:
            file.write(workflow_link)
        store_checksums(workflow_link_file)

    # TODO: remove 'openvinotoolkit/openvino' condition when Tokenizers artifacts are stored in a separate structure
    if not error_found and os.getenv('GITHUB_REPOSITORY') == 'openvinotoolkit/openvino':
        latest_artifacts_for_branch = artifact_utils.get_latest_artifacts_link(storage_dir, args.storage_root,
                                                                               args.branch_name, args.event_name,
                                                                               args.product_name)
        # Overwrite path to "latest" built artifacts only if a given commit is the head of a given branch
        if args.event_name != 'pull_request' and args.commit_sha == os.getenv('GITHUB_SHA'):
            # TODO: lock to avoid corruption in case of a parallel build (unlikely event for now, but still)
            with open(latest_artifacts_for_branch, 'w') as file:
                file.write(str(storage.relative_to(storage_root)))

    logger.debug(f"Copying finished")
    (storage / 'copying_finished').touch()
    if error_found:
        sys.exit(1)


if __name__ == '__main__':
    main()
