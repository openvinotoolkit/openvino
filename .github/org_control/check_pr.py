# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Check GitHub PRs and set labels by type and categories, e.g. 'ExternalPR', 'category: ci'
"""

# pylint: disable=fixme,no-member

import re
import datetime
from argparse import ArgumentParser
from enum import Enum

import github_api
from configs import Config


class PrType(Enum):
    """Constants for type of GitHub pull request by author membership"""
    EXTERNAL = 'ExternalPR'
    INTEL = 'ExternalIntelPR'
    ORG = 'OpenvinoPR'
    BAD = 'BadPR'


def get_pr_labels(pull):
    """Gets PR labels as set"""
    pr_lables = set()
    for label in pull.labels:
        pr_lables.add(label.name)
    return pr_lables


def set_pr_labels(pull, labels):
    """Sets new PR labels (all previously set labels are removed)"""
    if not labels or Config().DRY_RUN:
        return
    print('Set PR labels:', labels)
    # set_labels() should accept list but fails with empty "AssertionError:"
    pull.set_labels(labels)


def add_pr_labels(pull, labels):
    """Adds PR labels"""
    if not labels or Config().DRY_RUN:
        return
    print('Add PR labels:', labels)
    for label in labels:
        pull.add_to_labels(label)


def get_pr_type_by_labels(pull):
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


def get_label_by_team_name_re(team_name):
    """Generates label by PR reviwer team name using regular expressions"""
    if 'admins' in team_name:
        return 'category: ci'
    re_compile_label = re.compile(rf'{Config().GITHUB_REPO}-(.+)-maintainers')
    re_label = re_compile_label.match(team_name)
    if re_label:
        return f'category: {re_label.group(1).strip()}'
    return None


def get_label_by_team_name_map(team_name):
    """Generates label by PR reviwer team name using config map"""
    return Config().TEAM_TO_LABEL.get(team_name)


def get_category_labels(pull):
    """Gets list of category labels by all PR reviwer teams"""
    labels = []
    pr_lables = get_pr_labels(pull)
    for reviewer_team in pull.get_review_requests()[1]:
        reviewer_label = get_label_by_team_name_map(reviewer_team.name)
        if reviewer_label and reviewer_label not in pr_lables:
            labels.append(reviewer_label)
    return labels


def get_pr_info_str(pull):
    """Gets info about PR using a few workarounds"""
    pr_title = pull.title.encode("ASCII", "ignore").decode()

    # Workaround for PyGithub issue: https://github.com/PyGithub/PyGithub/issues/512
    pr_created_at = pull.created_at.replace(tzinfo=datetime.timezone.utc).astimezone()

    return f'PR: {pull.number} - {pr_title} - Created: {pr_created_at} - ' \
           f'Labels: {get_pr_labels(pull)} - Type: {get_pr_type_by_labels(pull)}'


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cfg-file", metavar="PATH", default=Config.default_cfg_path,
                            help=f"Path to json configuration file, e.g. {Config.default_cfg_path}")
    arg_parser.add_argument("--pr", metavar="NUMBER",
                            help="Get GitHub pull request with the number")
    arg_parser.add_argument("--pr-state", default="open", choices=["open", "closed"],
                            help="Set GitHub pull request state")
    arg_parser.add_argument("--newer", metavar="MINUTES",
                            help="Get newly created GitHub pull request only")
    args, unknown_args = arg_parser.parse_known_args()

    Config(args.cfg_file, unknown_args)
    gh_api = github_api.GithubOrgApi()

    if args.pr:
        pulls = [gh_api.repo.get_pull(int(args.pr))]
    else:
        pulls = gh_api.repo.get_pulls(state=args.pr_state)
        print(f'\nPRs count ({args.pr_state}):', pulls.totalCount)

    if args.newer:
        pr_created_after = (datetime.datetime.now() -
                            datetime.timedelta(minutes=int(args.newer))).astimezone()
        print('Checking PRs created after:', pr_created_after)
    non_org_intel_pr_users = set()
    non_org_pr_users = set()
    for pull in pulls:
        pr_created_at = pull.created_at.replace(tzinfo=datetime.timezone.utc).astimezone()
        if args.newer and pr_created_at <= pr_created_after:
            print(f'\nIGNORE: {get_pr_info_str(pull)}')
            continue
        pr_type_by_labels = get_pr_type_by_labels(pull)
        add_labels = []
        print(f'\n{get_pr_info_str(pull)}', end='')

        # Checks PR source type
        if gh_api.is_org_user(pull.user):
            print(' - Org user')
        elif github_api.is_intel_email(pull.user.email) or \
             github_api.is_intel_company(pull.user.company):
            print(' - Non org user with Intel email or company')
            non_org_intel_pr_users.add(pull.user)
            if pr_type_by_labels is not PrType.INTEL:
                print(f'NO "{PrType.INTEL.value}" label: ', end='')
                github_api.print_users(pull.user)
                add_labels.append(PrType.INTEL.value)
        elif github_api.is_user_ignored(pull.user):
            print(' - IGNORED non org user with NO Intel email or company')
        else:
            print(' - Non org user with NO Intel email or company')
            non_org_pr_users.add(pull.user)
            if pr_type_by_labels is not PrType.EXTERNAL:
                print(f'NO "{PrType.EXTERNAL.value}" label: ', end='')
                github_api.print_users(pull.user)
                add_labels.append(PrType.EXTERNAL.value)

        add_labels += get_category_labels(pull)
        add_pr_labels(pull, add_labels)

    print('\nNon org user with Intel email or company:')
    github_api.print_users(non_org_intel_pr_users)
    print('\nNon org user with NO Intel email or company:')
    github_api.print_users(non_org_pr_users)


if __name__ == '__main__':
    main()
