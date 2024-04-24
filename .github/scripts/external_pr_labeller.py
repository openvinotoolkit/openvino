from github import Github, Auth
import os
import logging

import argparse


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
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(
        level=LOGLEVEL,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d-%Y %H:%M:%S',
    )


if __name__ == '__main__':

    init_logger()

    LOGGER = logging.getLogger('labeller')
    EXTERNAL_PR_LABEL_NAME = 'ExternalPR'

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
            if label.name == EXTERNAL_PR_LABEL_NAME:
                LOGGER.info(f'THE PR ALREADY HAS THE "{EXTERNAL_PR_LABEL_NAME}" LABEL')
                break
        else:
            pr.add_to_labels(EXTERNAL_PR_LABEL_NAME)
            LOGGER.info(f'THE "{EXTERNAL_PR_LABEL_NAME}" LABEL WAS ADDED TO THE PR')
    else:
        LOGGER.info(
            f'THE {pr.user.login} IS A COLLABORATOR, NO NEED TO ADD THE "{EXTERNAL_PR_LABEL_NAME}" LABEL'
        )

    github.close()
