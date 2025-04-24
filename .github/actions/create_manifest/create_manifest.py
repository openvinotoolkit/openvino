from __future__ import annotations

import argparse
import logging
import os
from datetime import timezone
from pathlib import Path
import re
import git
import sys

from manifest_manager import Manifest, Repository, Component

sys.path.append(str(Path(__file__).parents[1]))
from common import artifact_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Creates manifest with product and repositories version')
    parser.add_argument('-e', '--event_name', help='Name of GitHub event', required=False)
    parser.add_argument('-r', '--repos', type=str, help='Paths to repositories to lon in manifest',
                        required=True)
    parser.add_argument('--product_type', help='Unique string to reflect product configuration', required=True)
    parser.add_argument('--target_arch', help='Target architecture', required=True)
    parser.add_argument('--build_type', help='Build type: release | debug | release_with_debug', required=True)
    parser.add_argument('--save_to', help='Path to save manifest to', required=True)
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


def get_repo_data(repo_dir: str | Path) -> dict:
    repo = git.Repo(str(repo_dir))
    repo_url = next(repo.remote().urls)
    repo_name_match = re.search(r'github\.com/[^/]+/([^/]+)', repo_url)
    repo_name = repo_name_match.group(1) if repo_name_match else None

    trigger_repo_url = f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
    is_trigger_repo = repo_url == trigger_repo_url

    branch = os.getenv('GITHUB_REF') if is_trigger_repo else repo.references[0].name
    target_branch = os.getenv('GITHUB_BASE_REF') if is_trigger_repo else None
    revision = os.getenv('TRIGGER_REPO_SHA') if is_trigger_repo else repo.head.commit.hexsha
    target_revision = os.getenv('BASE_SHA') if is_trigger_repo else None
    # Commit time of a merge commit (in case of PR merged to target)
    # TODO: Save commit time of a head commit in PR as well?
    commit_time = repo.head.commit.committed_datetime.astimezone(timezone.utc)
    merge_target = branch.endswith('/merge')
    return {
        'name': repo_name,
        'url': repo_url,
        'branch': branch.replace('refs/heads/', ''),  # To align with internal manifest
        'target_branch': target_branch,
        'revision': revision,
        'target_revision': target_revision,
        'commit_time': commit_time,
        'merge_target': merge_target,
        'trigger': is_trigger_repo,
    }


def parse_ov_version(header_file: str | Path) -> str:
    header_code = Path(header_file).read_text()
    major, minor, patch = (re.search(rf"#define OPENVINO_VERSION_{name} (\d+)", header_code).group(1)
                           for name in ["MAJOR", "MINOR", "PATCH"])
    return f"{major}.{minor}.{patch}"


def generate_manifest(repos: list, product_type: str, event_type: str, build_type: str, target_arch: str) -> Manifest:
    manifest = Manifest()
    component_name = None
    repositories = []
    ov_version = None
    trigger_repo = None

    for repo_dir in repos:
        repo = Repository(**get_repo_data(repo_dir))
        repositories.append(repo)
        if repo.name == 'openvino':
            version_file = Path(repo_dir) / 'src' / 'core' / 'include' / 'openvino' / 'core' / 'version.hpp'
            ov_version = parse_ov_version(version_file)
        if repo.trigger:
            trigger_repo = repo
            component_name = repo.name

    custom_branch_name = f'-{trigger_repo.branch}' if trigger_repo.branch != 'master' else ''
    run_number_postfix = f'-{os.environ.get("GITHUB_RUN_NUMBER")}' if os.environ.get("GITHUB_RUN_NUMBER") else ''
    product_version = f"{ov_version}{run_number_postfix}-{trigger_repo.revision[:11]}{custom_branch_name}"

    merge_queue_target_branch = next(iter(re.findall(f'^gh-readonly-queue/(.*)/', trigger_repo.branch)), None)
    target_branch = merge_queue_target_branch or trigger_repo.target_branch or trigger_repo.branch
    is_release_branch = re.match('^releases/.+$', target_branch)
    ci_build_dev_tag = f'dev{trigger_repo.commit_time.strftime("%Y%m%d")}' if not is_release_branch else ''
    wheel_product_version = f'{ov_version}.{ci_build_dev_tag}' if not is_release_branch else ov_version

    set_github_output('CI_BUILD_NUMBER', product_version, 'GITHUB_ENV')
    set_github_output('CI_BUILD_DEV_TAG', ci_build_dev_tag, 'GITHUB_ENV')

    component = Component(name=component_name, version=product_version, product_type=product_type,
                          target_arch=target_arch, build_type=build_type, build_event=event_type,
                          repositories=repositories, custom_params={'wheel_product_version': wheel_product_version})

    manifest.add_component(component)
    return manifest


def main():
    init_logger()
    logger = logging.getLogger(__name__)
    args = parse_args()

    event_name = args.event_name or os.getenv('GITHUB_EVENT_NAME')
    event_type = artifact_utils.get_event_type(event_name)

    repos = args.repos.split()
    manifest = generate_manifest(repos, args.product_type, event_type, args.build_type, args.target_arch)

    logger.info(f"Saving manifest to {args.save_to}")
    manifest.save_manifest(args.save_to)


if __name__ == '__main__':
    main()
