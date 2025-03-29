# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path, PureWindowsPath

sys.path.append(str(Path(__file__).parents[1]))
from common import artifact_utils, action_utils
from common.constants import PlatformMapping, PlatformKey

def parse_args():
    parser = argparse.ArgumentParser(description='Returns a path to artifacts for a given revision on a shared drive')
    artifact_utils.add_common_args(parser)
    parser.add_argument('-t', '--target_dir', type=str, help='Path to a dir in a workspace to download artifacts into',
                        required=True)
    parser.add_argument('-r', '--to_restore', type=str, required=False,
                        help='Comma-separated list of packages to restore, all available by default')
    parser.add_argument('-u', '--unpack_archives', action='store_true', required=False,
                        help='Whether to unpack all artifact archives once retrieved')
    args = parser.parse_args()
    return args


def include_filter(include_list: set | list):
    """
    Returns input for shutil.copytree ignore - to copy only files from include list
    """
    def _filter(root, files: list):
        if not include_list:
            return []
        return [f for f in files if f not in include_list and Path(root).name not in include_list]

    return _filter


def main():
    action_utils.init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()

    storage_dir = args.storage_dir or PlatformMapping[PlatformKey[args.platform.upper()]].value

    if args.commit_sha == 'latest_available_commit':
        latest_artifacts_link = artifact_utils.get_latest_artifacts_link(storage_dir, args.storage_root,
                                                                         args.branch_name, args.event_name,
                                                                         args.product_name)
        latest_artifacts_path = PureWindowsPath(latest_artifacts_link.read_text())
        normalized_path = latest_artifacts_path.as_posix() if os.name == 'posix' else latest_artifacts_path
        storage = Path(args.storage_root) / normalized_path
    else:
        storage = artifact_utils.get_storage_dir(storage_dir, args.commit_sha, args.storage_root, args.branch_name,
                                                 args.event_name, args.product_name)

    action_utils.set_github_output("artifacts_storage_path", str(storage))
    logger.info(f"Artifacts are taken from here: {storage}")

    main_package_extension = 'zip' if 'windows' in storage_dir else 'tar.gz'
    main_package_name = f'openvino_package.{main_package_extension}'
    defaults = [main_package_name, 'manifest.yml']
    to_restore = set(args.to_restore.split(',')).union(defaults) if args.to_restore else defaults
    if args.to_restore == 'all':
        to_restore = None
    shutil.copytree(storage, args.target_dir, dirs_exist_ok=True,
                    ignore=include_filter(to_restore))
    logger.info(f"Artifacts are copied here: {args.target_dir}")

    if args.unpack_archives:
        for file in Path(args.target_dir).iterdir():
            logger.info(f"file: {file}")
            if not file.is_file() or file.is_file() and not (tarfile.is_tarfile(file) or zipfile.is_zipfile(file)):
                continue
            output_dir = Path(args.target_dir) / Path(str(file).removesuffix('.tar.gz')).stem
            logger.info(f"Unpacking: {file} to {output_dir}")
            shutil.unpack_archive(file, output_dir)
            file.unlink()

    action_utils.set_github_output("artifacts_path", args.target_dir)


if __name__ == '__main__':
    main()
