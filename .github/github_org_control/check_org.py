# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Check GitHub organization and invite members
"""

# pylint: disable=fixme,no-member,too-many-locals

import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[1]))
from github_org_control.configs import Config
from github_org_control.github_api import GithubOrgApi, get_dev_emails, print_users
from github_org_control.ldap_api import LdapApi, print_user_info, InfoLevel


def remove_members(gh_api, cfg_emails, org_emails, dev_emails, org_emails_no_in_ldap):
    """Checks and remove members"""
    print(
        f"\n{'=' * 10} Check accounts below and remove from the GitHub organization or "
        f"configuration {'=' * 10}"
    )

    cfg_emails_no_in_org = sorted(cfg_emails.difference(org_emails))
    print(
        f"\nCfg developer emails - absent in GitHub organization {len(cfg_emails_no_in_org)}:",
        "; ".join(cfg_emails_no_in_org),
    )

    non_member_ignored_logins = set(Config().IGNORE_LOGINS).difference(
        set(gh_api.org_members_by_login.keys())
    )
    print(
        f"\nIgnored logins - absent in GitHub organization {len(non_member_ignored_logins)}:\n",
        "\n".join(non_member_ignored_logins),
    )

    org_emails_no_in_dev = sorted(org_emails.difference(dev_emails))
    print(
        f"\nOrg member emails - absent in cfg and LDAP PDLs {len(org_emails_no_in_dev)}:",
        "; ".join(org_emails_no_in_dev),
    )

    print(
        f"\nOrg member emails - absent in LDAP at all {len(org_emails_no_in_ldap)}:",
        "; ".join(sorted(org_emails_no_in_ldap)),
    )

    print("\nOrg members - no real name:")
    members_to_fix_name = sorted(gh_api.members_to_fix_name, key=lambda member: member.email)
    print_users(members_to_fix_name)
    print(
        "\nOrg member emails - no real name:",
        "; ".join([member.email.lower() for member in members_to_fix_name]),
    )

    print("\nOrg members - no Intel emails:")
    print_users(gh_api.members_to_remove)

    gh_api.remove_users(org_emails_no_in_ldap | gh_api.members_to_remove)


def main():
    """The main entry point function"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--cfg-file",
        metavar="PATH",
        default=Config.default_cfg_path,
        help=f"Path to json configuration file, e.g. {Config.default_cfg_path}",
    )
    arg_parser.add_argument("--teams", action="store_true", help="Check GitHub teams")
    arg_parser.add_argument("--no-ldap", action="store_true", help="Don't use LDAP info")
    args, unknown_args = arg_parser.parse_known_args()

    Config(args.cfg_file, unknown_args)
    gh_api = GithubOrgApi()

    if args.teams:
        gh_api.get_org_teams()
        return

    cfg_emails = get_dev_emails()
    print(f"\nCfg developer emails {len(cfg_emails)}:", "; ".join(sorted(cfg_emails)))

    dev_emails = set()
    dev_emails.update(cfg_emails)

    if not args.no_ldap:
        ldap_api = LdapApi()
        ldap_emails = ldap_api.get_user_emails()
        dev_emails.update(ldap_emails)
        print(f"\nLDAP developer emails {len(ldap_emails)}:", "; ".join(sorted(ldap_emails)))

        cfg_emails_no_in_ldap = ldap_api.get_absent_emails(cfg_emails)
        print(
            f"\nCfg developer emails - absent in LDAP at all {len(cfg_emails_no_in_ldap)}:",
            "; ".join(sorted(cfg_emails_no_in_ldap)),
        )

        cfg_ldap_inters = cfg_emails.intersection(ldap_emails)
        print(
            f"\nCfg developer emails - present in LDAP developers {len(cfg_ldap_inters)}:",
            "; ".join(sorted(cfg_ldap_inters)),
        )

    org_emails = gh_api.get_org_emails()
    print(f"\nOrg emails {len(org_emails)}:", "; ".join(sorted(org_emails)))

    org_emails_no_in_ldap = set()
    if not args.no_ldap:
        org_ldap_diff = org_emails.difference(ldap_emails)
        print(
            f"\nOrg member emails - absent in LDAP developers {len(org_ldap_diff)}:",
            "; ".join(sorted(org_ldap_diff)),
        )

        for email in org_ldap_diff:
            user_info = ldap_api.get_user_info_by_email(email)
            if user_info:
                print_user_info(user_info, InfoLevel.PDL)
            else:
                org_emails_no_in_ldap.add(email)

    org_pendig_invitation_emails = gh_api.get_org_invitation_emails()
    invite_emails = dev_emails.difference(org_emails).difference(org_pendig_invitation_emails)
    print(f"\nInvite emails {len(invite_emails)}:", "; ".join(sorted(invite_emails)))

    valid_github_users = gh_api.get_valid_github_users(invite_emails)
    gh_api.invite_users(valid_github_users)

    remove_members(gh_api, cfg_emails, org_emails, dev_emails, org_emails_no_in_ldap)


if __name__ == "__main__":
    main()
