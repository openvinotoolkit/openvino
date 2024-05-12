# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
from enum import Enum

from github import Github, Auth


class ExternalPRLabels(str, Enum):
    ExternalPR = 'ExternalPR'
    ExternalIntelPR = 'ExternalIntelPR'


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--repository-name',
        type=str,
        required=True,
        help='Repository name in the OWNER/REPOSITORY format',
    )
    parser.add_argument(
        '--pr-number', type=int, required=True, help='PR number to label'
    )
    return parser.parse_args()


def init_logger():
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d-%Y %H:%M:%S',
    )


if __name__ == '__main__':

    init_logger()

    LOGGER = logging.getLogger('labeller')

    args = get_arguments()
    pr_number = args.pr_number
    repository_name = args.repository_name

    github = Github(auth=Auth.Token(token=os.environ.get('GITHUB_TOKEN')))
    gh_repo = github.get_repo(full_name_or_id=repository_name)

    pr = gh_repo.get_pull(number=pr_number)

    LOGGER.info(f'CONTEXT: PR #{pr_number}. USER: {pr.user.login}. ALL PR LABELS: {list(pr.get_labels())}')

    if not gh_repo.has_in_collaborators(pr.user.login):
        LOGGER.info(f'THE {pr.user.login} IS NOT A COLLABORATOR')

        for label in pr.get_labels():
            if label.name in (ExternalPRLabels.ExternalPR, ExternalPRLabels.ExternalIntelPR):
                LOGGER.info(f'THE PR ALREADY HAS THE "{label.name}" LABEL')
                break
        else:
            is_intel_user = bool(pr.user.email and pr.user.email.endswith('@intel.com'))
            label_to_add: str = ExternalPRLabels.ExternalIntelPR.name if is_intel_user else ExternalPRLabels.ExternalPR.name

            pr.add_to_labels(label_to_add)
            LOGGER.info(f'THE "{label_to_add}" LABEL WAS ADDED TO THE PR')
    else:
        LOGGER.info(
            f'THE {pr.user.login} IS A COLLABORATOR, NO NEED TO ADD THE "External" LABEL'
        )

    github.close()
