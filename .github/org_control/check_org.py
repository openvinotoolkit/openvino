# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Check GitHub organization and invite members
"""

# pylint: disable=fixme,no-member

from argparse import ArgumentParser

import github_api
from configs import Config


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cfg-file", metavar="PATH", default=Config.default_cfg_path,
                            help=f"Path to json configuration file, e.g. {Config.default_cfg_path}")
    arg_parser.add_argument("--teams", action="store_true", help="Check GitHub teams")
    args, unknown_args = arg_parser.parse_known_args()

    Config(args.cfg_file, unknown_args)
    gh_api = github_api.GithubOrgApi()

    if args.teams:
        gh_api.get_org_teams()
    else:
        dev_emails = github_api.get_dev_emails()
        print(f'\nDeveloper emails {len(dev_emails)}:', '; '.join(dev_emails))

        org_emails = gh_api.get_org_emails()
        print(f'\nOrg emails {len(org_emails)}:', '; '.join(org_emails))

        org_pendig_invitation_emails = gh_api.get_org_invitation_emails()

        invite_emails = dev_emails.difference(org_emails).difference(org_pendig_invitation_emails)
        print(f'\nInvite emails {len(invite_emails)}:', '; '.join(invite_emails))

        no_in_dev_emails = org_emails.difference(dev_emails)
        print(f'\nOrg members - no in developers list {len(no_in_dev_emails)}:',
              '; '.join(no_in_dev_emails))

        valid_github_users = gh_api.get_valid_github_users(invite_emails)

        gh_api.invite_users(valid_github_users)


if __name__ == '__main__':
    main()
