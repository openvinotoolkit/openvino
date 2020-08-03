# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Check GitHub organization and invite members
"""

# pylint: disable=fixme,no-member,too-many-locals

from argparse import ArgumentParser

from configs import Config
from github_api import GithubOrgApi, get_dev_emails
from ldap_api import LdapApi, print_user_info, InfoLevel


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cfg-file", metavar="PATH", default=Config.default_cfg_path,
                            help=f"Path to json configuration file, e.g. {Config.default_cfg_path}")
    arg_parser.add_argument("--teams", action="store_true", help="Check GitHub teams")
    arg_parser.add_argument("--no-ldap", action="store_true", help="Don't use LDAP info")
    args, unknown_args = arg_parser.parse_known_args()

    Config(args.cfg_file, unknown_args)
    gh_api = GithubOrgApi()

    if args.teams:
        gh_api.get_org_teams()
        return

    cfg_emails = get_dev_emails()
    print(f'\nDeveloper cfg emails {len(cfg_emails)}:', '; '.join(cfg_emails))

    if not args.no_ldap:
        ldap_api = LdapApi()
        absent_emails = ldap_api.get_absent_emails(cfg_emails)
        print(f'\nDeveloper cfg emails - absent in LDAP {len(absent_emails)}:',
              '; '.join(absent_emails))

        ldap_emails = ldap_api.get_user_emails()
        print(f'\nDeveloper LDAP emails {len(ldap_emails)}:', '; '.join(ldap_emails))

    org_emails = gh_api.get_org_emails()
    print(f'\nOrg emails {len(org_emails)}:', '; '.join(org_emails))

    dev_emails = cfg_emails
    if not args.no_ldap:
        ldap_cfg_diff = cfg_emails.difference(ldap_emails)
        print(f'\nCfg diff LDAP emails {len(ldap_cfg_diff)}:', '; '.join(ldap_cfg_diff))
        ldap_org_diff = org_emails.difference(ldap_emails)
        print(f'\nOrg diff LDAP emails {len(ldap_org_diff)}:', '; '.join(ldap_org_diff))
        no_in_ldap_emails = set()
        for email in ldap_org_diff:
            user_info = ldap_api.get_user_info_by_email(email)
            if user_info:
                print_user_info(user_info, InfoLevel.PDL)
            else:
                no_in_ldap_emails.add(email)
        print(f'\nOrg members - no in LDAP {len(no_in_ldap_emails)}:', '; '.join(no_in_ldap_emails))

        dev_emails.update(ldap_emails)

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
