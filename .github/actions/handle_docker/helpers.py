# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
from ghapi.all import GhApi
from pathlib import Path


def init_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')


def set_github_output(name: str, value: str, github_output_var_name: str = 'GITHUB_OUTPUT'):
    """Sets output variable for a GitHub Action"""
    logger = logging.getLogger(__name__)
    # In an environment variable "GITHUB_OUTPUT" GHA stores path to a file to write outputs to
    with open(os.environ.get(github_output_var_name), 'a+') as file:
        logger.info(f"Add {name}={value} to {github_output_var_name}")
        print(f'{name}={value}', file=file)


def images_to_output(images: list):
    images_output = {}
    for image in images:
        image_name, os_name = image.name.split('/', 1)
        if image_name not in images_output:
            images_output[image_name] = {}

        images_output[image_name][os_name] = image.ref()

    return images_output


def get_changeset(repo: str, pr: str, target_branch: str, commit_sha: str):
    """Returns changeset either from PR or commit"""
    owner, repository = repo.split('/')
    gh_api = GhApi(owner=owner, repo=repository, token=os.getenv("GITHUB_TOKEN"))
    if pr:
        changed_files = gh_api.pulls.list_files(pr)
    elif target_branch:
        target_branch_head_commit = gh_api.repos.get_branch(target_branch).commit.sha
        changed_files = gh_api.repos.compare_commits(f'{target_branch_head_commit}...{commit_sha}').get('files', [])
    else:
        raise ValueError(f'Either "pr" or "target_branch" parameter must be non-empty')
    return set([f.filename for f in changed_files])


def run(cmd: str, dry_run: bool = False, fail_on_error: bool = True):
    logger = logging.getLogger('run')
    logger.info(cmd)

    if dry_run:
        return 0, ''

    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            logger.info(line.strip())

        proc.communicate()
        if proc.returncode != 0:
            msg = f"Command '{cmd}' returned non-zero exit status {proc.returncode}"
            if fail_on_error:
                raise RuntimeError(msg)

            logger.warning(msg)
            return proc.returncode


def name_from_dockerfile(dockerfile: str | Path, dockerfiles_root: str | Path) -> str:
    image_name = str(Path(dockerfile).relative_to(dockerfiles_root).parent.as_posix())
    return image_name
