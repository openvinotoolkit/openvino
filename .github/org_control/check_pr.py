# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Check GitHub PRs and set labels by type and categories, e.g. 'ExternalPR', 'category: ci'
"""

# pylint: disable=fixme,no-member

import re
from argparse import ArgumentParser
from enum import Enum

import github_api
from configs import Config


class PrType(Enum):
    """Constants for type of GitHub pull request by author membership"""
    EXTERNAL = 'ExternalPR'
    INTEL = 'IntelDevPR'
    ORG = 'OpenvinoDevPR'
    BAD = 'BadPR'


def get_pr_labels(pull):
    """Gets PR labels as set"""
    pr_lables = set()
    for label in pull.labels:
        pr_lables.add(label.name)
    return pr_lables


def set_pr_label(pull, labels):
    """Sets PR labels"""
    if not labels or Config().DRY_RUN:
        return
    print(f'Set PR labels:', labels)
    # TODO: Review labels and enable. Check setting existing labels
    #pull.set_labels(labels)


def get_pr_type(pull):
    """Gets PR type using labels"""
    pr_lables = get_pr_labels(pull)
    pr_types = set(type.value for type in PrType)
    pr_types_labels = pr_lables & pr_types
    if not pr_types_labels:
        return None
    if len(pr_types_labels) > 1:
        print(f'Duplicated labels: {pr_types_labels}')
        return PrType.BAD
    return PrType(PrType(pr_types_labels.pop()))


def get_label_by_team_name(team_name):
    """Generates label by PR reviwer team name"""
    if 'admins' in team_name:
        return 'category: ci'
    re_compile_label = re.compile(rf'{Config().GITHUB_REPO}-(.+)-maintainers')
    re_label = re_compile_label.match(team_name)
    if re_label:
        return f'category: {re_label.group(1).strip()}'
    return None


def get_category_labels(pull):
    """Gets list of category labels by all PR reviwer teams"""
    labels = []
    pr_lables = get_pr_labels(pull)
    for reviewer_team in pull.get_review_requests()[1]:
        reviewer_label = get_label_by_team_name(reviewer_team.name)
        if reviewer_label and reviewer_label not in pr_lables:
            labels.append(reviewer_label)
    return labels


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cfg-file", metavar="PATH", default=Config.default_cfg_path,
                            help=f"Path to json configuration file, e.g. {Config.default_cfg_path}")
    arg_parser.add_argument("--pr", metavar="NUMBER",
                            help="Get GitHub pull request with the number")
    arg_parser.add_argument("--pr-state", default="open", choices=["open", "closed"],
                            help="Set GitHub pull request state")
    args, unknown_args = arg_parser.parse_known_args()

    Config(args.cfg_file, unknown_args)
    gh_api = github_api.GithubOrgApi()

    if args.pr:
        pulls = [gh_api.repo.get_pull(int(args.pr))]
    else:
        pulls = gh_api.repo.get_pulls(state=args.pr_state)
        print(f'PRs count ({args.pr_state}):', pulls.totalCount)
    non_org_intel_pr_users = set()
    non_org_pr_users = set()
    for pull in pulls:
        pr_lables = get_pr_labels(pull)
        pr_type = get_pr_type(pull)
        set_labels = []
        print('\n', pull, f'- Labels: {pr_lables} -', f'Type: {pr_type}', end='')

        # Checks PR source type
        if gh_api.is_org_user(pull.user):
            print(' - Org user')
            if pr_type is not PrType.ORG:
                print(f'NO "{PrType.ORG.value}" label: ', end='')
                github_api.print_users(pull.user)
                set_labels.append(PrType.ORG.value)
        elif github_api.is_intel_email(pull.user.email) or \
             github_api.is_intel_company(pull.user.company):
            print(' - Non org user with Intel email or company')
            non_org_intel_pr_users.add(pull.user)
            if pr_type is not PrType.INTEL:
                print(f'NO "{PrType.INTEL.value}" label: ', end='')
                github_api.print_users(pull.user)
                set_labels.append(PrType.INTEL.value)
        else:
            print(f' - Non org user with NO Intel email or company')
            non_org_pr_users.add(pull.user)
            if pr_type is not PrType.EXTERNAL:
                print(f'NO "{PrType.EXTERNAL.value}" label: ', end='')
                github_api.print_users(pull.user)
                set_labels.append(PrType.EXTERNAL.value)

        set_labels += get_category_labels(pull)
        set_pr_label(pull, set_labels)

    print(f'\nNon org user with Intel email or company:')
    github_api.print_users(non_org_intel_pr_users)
    print(f'\nNon org user with NO Intel email or company:')
    github_api.print_users(non_org_pr_users)


if __name__ == '__main__':
    main()
