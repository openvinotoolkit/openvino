# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from common import artifact_utils, action_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Returns a path to artifacts for a given revision on a shared drive')
    artifact_utils.add_common_args(parser)
    parser.add_argument('-t', '--target_dir', type=str, help='Path to a dir in a workspace to download artifacts into',
                        required=True)
    parser.add_argument('-k', '--artifact_key', type=str,
                        help='A key under which to upload the artifacts to storage, product type by default',
                        required=False)
    args = parser.parse_args()
    return args


def main():
    action_utils.init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()

    if args.commit_sha == 'latest':
        latest_artifacts_link = artifact_utils.get_latest_artifacts_link(args.storage_dir, args.storage_root,
                                                                         args.branch_name, args.event_name)
        storage = Path(args.storage_root) / latest_artifacts_link.read_text()
    else:
        storage = artifact_utils.get_storage_dir(args.storage_dir, args.commit_sha, args.storage_root, args.branch_name,
                                                 args.event_name)

    action_utils.set_github_output("artifacts_storage_path", str(storage))
    logger.info(f"Artifacts are taken from here: {storage}")

    shutil.copytree(storage, args.target_dir, dirs_exist_ok=True)
    logger.info(f"Artifacts are copied here: {args.target_dir}")

    action_utils.set_github_output("artifacts_workspace_path", args.target_dir)
    action_utils.set_github_output("restored_artifacts_key", args.artifact_key or args.storage_dir)


if __name__ == '__main__':
    main()
