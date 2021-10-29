# Copyright (C) 2018-2021 Intel Corporation
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
    print(f'\nCfg developer emails {len(cfg_emails)}:', '; '.join(sorted(cfg_emails)))

    dev_emails = set()
    dev_emails.update(cfg_emails)

    if not args.no_ldap:
        ldap_api = LdapApi()
        ldap_emails = ldap_api.get_user_emails()
        dev_emails.update(ldap_emails)
        print(f'\nLDAP developer emails {len(ldap_emails)}:', '; '.join(sorted(ldap_emails)))

        cfg_emails_no_in_ldap = ldap_api.get_absent_emails(cfg_emails)
        print(f'\nCfg developer emails - absent in LDAP at all {len(cfg_emails_no_in_ldap)}:',
              '; '.join(sorted(cfg_emails_no_in_ldap)))

        cfg_ldap_inters = cfg_emails.intersection(ldap_emails)
        print(f'\nCfg developer emails - present in LDAP developers {len(cfg_ldap_inters)}:',
              '; '.join(sorted(cfg_ldap_inters)))

    org_emails, org_logins_no_intel_email = gh_api.get_org_emails()
    print(f'\nOrg emails {len(org_emails)}:', '; '.join(sorted(org_emails)))

    org_emails_no_in_ldap = set()
    if not args.no_ldap:
        org_ldap_diff = org_emails.difference(ldap_emails)
        print(f'\nOrg member emails - absent in LDAP developers {len(org_ldap_diff)}:',
              '; '.join(sorted(org_ldap_diff)))

        for email in org_ldap_diff:
            user_info = ldap_api.get_user_info_by_email(email)
            if user_info:
                print_user_info(user_info, InfoLevel.PDL)
            else:
                org_emails_no_in_ldap.add(email)

    org_pendig_invitation_emails = gh_api.get_org_invitation_emails()
    invite_emails = dev_emails.difference(org_emails).difference(org_pendig_invitation_emails)
    print(f'\nInvite emails {len(invite_emails)}:', '; '.join(sorted(invite_emails)))

    valid_github_users = gh_api.get_valid_github_users(invite_emails)
    gh_api.invite_users(valid_github_users)

    print('\nCheck accounts below and remove from the GitHub organization and cfg list')

    cfg_emails_no_in_org = sorted(cfg_emails.difference(org_emails))
    print(f'\nCfg developer emails - absent in GitHub organization {len(cfg_emails_no_in_org)}:',
          '; '.join(cfg_emails_no_in_org))

    org_emails_no_in_dev = sorted(org_emails.difference(dev_emails))
    print(f'\nOrg member emails - absent in cfg and LDAP developers {len(org_emails_no_in_dev)}:',
          '; '.join(org_emails_no_in_dev))

    print(f'\nOrg member emails - absent in LDAP at all {len(org_emails_no_in_ldap)}:',
          '; '.join(sorted(org_emails_no_in_ldap)))

    print(f'\nOrg member logins - absent Intel email {len(org_logins_no_intel_email)}:',
          '; '.join(sorted(org_logins_no_intel_email)))


if __name__ == '__main__':
    main()
