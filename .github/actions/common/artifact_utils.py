from __future__ import annotations

import argparse
import os
from pathlib import Path
from .constants import EventType, ProductType, PlatformKey


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('-s', '--commit_sha', help='Commit hash for which artifacts were generated', required=True)
    parser.add_argument('-b', '--branch_name', help='Name of GitHub branch', required=False,
                        default=os.getenv('GITHUB_BASE_REF') or
                                os.getenv('MERGE_QUEUE_BASE_REF').replace('refs/heads/', '') or
                                os.getenv('GITHUB_REF_NAME'))
    parser.add_argument('-e', '--event_name', help='Name of GitHub event', required=False,
                        default=os.getenv('GITHUB_EVENT_NAME'))
    parser.add_argument('--storage_root', help='Root path of the artifacts storage', required=False,
                        default=os.getenv('ARTIFACTS_SHARE'))
    parser.add_argument('-n', '--product_name', required=False, default='dldt',
                        help='Product name for which artifacts are generated')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--storage_dir', help='Subdirectory name for artifacts, same as product type',
                       choices=[product_type.value for product_type in ProductType], type=str.lower)
    group.add_argument('-p', '--platform', type=str.lower,
                       help='Platform for which to restore artifacts. Used if storage_dir is not set',
                       choices=[platform_key.value for platform_key in PlatformKey])


def get_event_type(event_name: str = os.getenv('GITHUB_EVENT_NAME')) -> str:
    return EventType.pre_commit.value if event_name == 'pull_request' else EventType.commit.value


def get_storage_event_dir(storage_root: str | Path, branch_name: str, event_name: str,
                          product_name: str = 'dldt') -> Path:
    """ Returns path to stored artifacts for a given branch and event """
    event_type = get_event_type(event_name)
    storage_branch_dir = Path(storage_root) / product_name / branch_name / event_type
    return storage_branch_dir


def get_storage_dir(product_type: str, commit_hash: str, storage_root: str | Path, branch_name: str, event_name: str,
                    product_name: str = 'dldt') -> Path:
    """ Returns full path to stored artifacts for a given product type """

    # TODO: return, once we decide to get rid of post-commit and choose artifacts generated for a merged PR in queue?
    # merge_queue_matcher = re.search(r'gh-readonly-queue/(.*?)/pr-', branch_name)
    # if merge_queue_matcher:
    #     branch_name = merge_queue_matcher.group(1)

    storage_event_dir = get_storage_event_dir(storage_root, branch_name, event_name, product_name)
    storage = storage_event_dir / commit_hash / product_type.lower()
    return storage


def get_latest_artifacts_link(product_type: str, storage_root: str | Path, branch_name: str, event_name: str,
                              product_name: str = 'dldt') -> Path:
    """ Returns path to latest available artifacts for a given branch, event and product type """
    storage_branch_dir = get_storage_event_dir(storage_root, branch_name, event_name, product_name)
    latest_artifacts_for_branch = storage_branch_dir / f"latest_{product_type}.txt"
    return Path(latest_artifacts_for_branch)
