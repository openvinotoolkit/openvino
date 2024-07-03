import argparse
import logging
import os
import re
import sys
import shutil
from contextlib import contextmanager
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Returns product components changed in a given PR or commit')
    parser.add_argument('-e', '--event_name', help='Name of GitHub event', required=False)
    parser.add_argument('-b', '--branch_name', help='Name of GitHub branch', required=False)
    parser.add_argument('-s', '--commit_sha', help='Commit hash for which artifacts were generated', required=False)
    parser.add_argument('-a', '--artifacts', type=str, help='Paths to artifacts to store (files/dirs)', required=True)
    parser.add_argument('--storage_dir', help='Directory name to store artifacts in', required=True)
    parser.add_argument('--storage_root', help='Root path of the storage to place artifacts to', required=True)
    args = parser.parse_args()
    return args


def init_logger():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


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

    github_server = os.getenv('GITHUB_SERVER_URL')
    if github_server:  # If running from GHA context
        repository = os.getenv('GITHUB_REPOSITORY')
        run_id = os.getenv('GITHUB_RUN_ID')
        # TODO: write an exact job link, but it's not trivial to get
        workflow_link = f"{github_server}/{repository}/actions/runs/{run_id}"
        with open(storage / 'workflow_link.txt', 'w') as file:
            file.write(workflow_link)

    logger.debug(f"Copying finished")
    (storage / 'copying_finished').touch()
    if error_found:
        sys.exit(1)


if __name__ == '__main__':
    main()
