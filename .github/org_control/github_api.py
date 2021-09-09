# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GitHub API for controlling organization
"""

# pylint: disable=fixme,no-member

import re
import time

from github import Github, GithubException, RateLimitExceededException, IncompletableObject
from github.PaginatedList import PaginatedList

from configs import Config


def is_valid_user(user):
    """Checks that user is valid github.Github object"""
    try:
        return user and user.login
    except IncompletableObject:
        return False


def is_user_ignored(user):
    """Checks that user should be ignored"""
    cfg = Config()
    if is_valid_user(user) and user.login.lower() not in cfg.properties['IGNORE_LOGINS']:
        return False
    return True


def is_valid_name(name):
    """Checks that GitHub user's name is valid"""
    return name and len(name) >= 3 and ' ' in name


def is_intel_email(email):
    """Checks that email is valid Intel email"""
    return email and len(email) > 10 and ' ' not in email and email.lower().endswith('@intel.com')


def is_intel_company(company):
    """Checks that company contains intel"""
    return company and 'intel' in company.lower()


def is_valid_intel_user(user):
    """Checks that user is valid GitHub and Intel user"""
    return is_valid_user(user) and (is_valid_name(user.name) and is_intel_email(user.email) or
                                    is_user_ignored(user))


def print_users(users):
    """Print list of users in different formats: list, set, PaginatedList"""
    if isinstance(users, (list, set, PaginatedList)):
        users_count = users.totalCount if isinstance(users, PaginatedList) else len(users)
        print(f'\nGitHub users {users_count} (login - name - company - email - valid):')
    else:
        users = [users]
    for user in users:
        if not is_valid_user(user):
            print('WRONG GitHub user: ???')
            continue
        valid_check = 'OK' if is_valid_intel_user(user) else 'FIX'
        if not is_intel_email(user.email):
            valid_check += ' email'
        if not is_valid_name(user.name):
            valid_check += ' name'
        print(f'{user.login} - "{user.name}" - "{user.company}" - {user.email} - {valid_check}')


def get_dev_emails():
    """
    Read a file with developer emails. Supported email formats
    first_name.last_name@intel.com
    Import from Outlook: Last_name, First_name <first_name.last_name@intel.com>
    """
    re_email = re.compile(r'.+<(.+)>')
    emails = set()
    cfg = Config()
    with open(cfg.properties['EMAILS_FILE_PATH']) as file_obj:
        for line in file_obj:
            line = line.strip().lower()
            if not line or line.startswith('#'):
                continue
            re_outlook_email = re_email.match(line)
            if re_outlook_email:
                line = re_outlook_email.group(1).strip()
            if not is_intel_email(line):
                print(f'Wrong email in {cfg.properties["EMAILS_FILE_PATH"]}: {line}')
                continue
            emails.add(line)
    return emails


class GithubOrgApi:
    """Common API for GitHub organization"""

    def __init__(self):
        self._cfg = Config()
        self.github = Github(self._cfg.GITHUB_TOKEN)
        self.github_org = self.github.get_organization(self._cfg.GITHUB_ORGANIZATION)
        self.repo = self.github.get_repo(f'{self._cfg.GITHUB_ORGANIZATION}/'
                                         f'{self._cfg.GITHUB_REPO}')

    def is_org_user(self, user):
        """Checks that user is a member of GitHub organization"""
        if is_valid_user(user):
            # user.get_organization_membership(self.github_org) doesn't work with org members
            # permissions, GITHUB_TOKEN must be org owner now
            return self.github_org.has_in_members(user)
        return False

    def get_org_emails(self):
        """Gets and prints emails of all GitHub organization members"""
        org_members = self.github_org.get_members()
        org_emails = set()
        org_members_fix = set()
        org_emails_fix_name = set()
        org_logins_fix_intel_email = set()

        print(f'\nOrg members {org_members.totalCount} (login - name - company - email - valid):')
        for org_member in org_members:
            print_users(org_member)
            if is_user_ignored(org_member):
                continue
            if is_intel_email(org_member.email):
                org_emails.add(org_member.email.lower())
                if not is_valid_name(org_member.name):
                    org_members_fix.add(org_member)
                    org_emails_fix_name.add(org_member.email.lower())
            else:
                org_members_fix.add(org_member)
                org_logins_fix_intel_email.add(org_member.login.lower())

        print_users(org_members_fix)
        print(f'\nOrg members - no Intel emails {len(org_logins_fix_intel_email)}:',
              '; '.join(org_logins_fix_intel_email))
        print(f'\nOrg members - no real name {len(org_emails_fix_name)}:',
              '; '.join(org_emails_fix_name))
        return (org_emails, org_logins_fix_intel_email)

    def get_org_invitation_emails(self):
        """Gets GitHub organization teams prints info"""
        org_invitations = self.github_org.invitations()
        org_invitation_emails = set()

        print(f'\nOrg invitations {org_invitations.totalCount} (login - name - email - valid):')
        for org_invitation in org_invitations:
            # TODO: investigate GithubException while access to user name and enable print_users()
            # github.GithubException.IncompletableObject: 400 "Returned object contains no URL"
            #print_users(org_invitation)
            print(f'{org_invitation.login} - ??? - {org_invitation.email} - ???')
            if is_user_ignored(org_invitation):
                continue
            if is_intel_email(org_invitation.email):
                org_invitation_emails.add(org_invitation.email.lower())
            else:
                print('Strange org invitation:', org_invitation)

        print(f'\nOrg invitation emails {len(org_invitation_emails)}:',
              '; '.join(org_invitation_emails))
        return org_invitation_emails

    def get_org_teams(self):
        """Gets GitHub organization teams prints info"""
        teams = []
        org_teams = self.github_org.get_teams()
        print('\nOrg teams count:', org_teams.totalCount)
        for team in org_teams:
            teams.append(team.name)
            print(f'\nTeam: {team.name} - parent: {team.parent}')

            repos = team.get_repos()
            print('Repos:')
            for repo in repos:
                print(f'    {repo.name} -', team.get_repo_permission(repo))

            team_maintainers = team.get_members(role='maintainer')
            team_maintainer_logins = set()
            for maintainer in team_maintainers:
                team_maintainer_logins.add(maintainer.login)
            team_members = team.get_members(role='member')
            team_member_logins = set()
            for member in team_members:
                team_member_logins.add(member.login)
            members = team.get_members(role='all')
            member_emails = []
            print('Members (role - login - name - company - email - valid):')
            for user in members:
                if user.login in team_maintainer_logins:
                    print('    Maintainer - ', end='')
                elif user.login in team_member_logins:
                    print('    Member - ', end='')
                else:
                    # It is not possible to check child teams members
                    print('    ??? - ', end='')
                print_users(user)
                if is_intel_email(user.email) and not is_user_ignored(user):
                    member_emails.append(user.email.lower())
            print(f'Intel emails {len(member_emails)}:', '; '.join(member_emails))
        return teams

    def get_valid_github_users(self, emails):
        """Gets valid GitHub users by email and prints status"""
        valid_users = set()
        no_account_emails = set()
        print(f'\nGitHub users from {len(emails)} invite emails (email - status):')
        for email in emails:
            if not is_intel_email(email):
                print(f'{email} - Non Intel email')
                continue

            # You can make up to 30 requests per minute; https://developer.github.com/v3/search/
            # Sleep 2.4 sec is about 25 requests per minute
            time.sleep(2.4)
            try:
                users = self.github.search_users(f'{email} in:email')
            except RateLimitExceededException:
                time.sleep(5)
                users = self.github.search_users(f'{email} in:email')

            if users.totalCount == 0:
                print(f'{email} - No valid GitHub account')
                no_account_emails.add(email)
                continue
            if users.totalCount > 1:
                print(f'{email} - Found {users.totalCount} GitHub accounts')
            for user in users:
                if user.email and user.email.lower() == email:
                    print(f'{email} - OK')
                    valid_users.add(user)
                else:
                    print(f'{email} - Non public or wrong email - login: {user.login} - '
                          f'email: {user.email}')
        print('Valid users count:', len(valid_users))
        print_users(valid_users)
        print(f'\nIntel emails - No valid GitHub account {len(no_account_emails)}:',
              '; '.join(no_account_emails))
        return valid_users

    def invite_users(self, users):
        """Invites users and prints status"""
        if isinstance(users, (list, set)):
            print(f'\nInvite {len(users)} users:')
        else:
            users = [users]

        for user in users:
            if isinstance(user, str):
                print(f'Email: {user}')
                self.github_org.invite_user(email=user)
            else:
                print(f'{user.login} - "{user.name}" - {user.email} - ', end='')
                try:
                    if is_user_ignored(user):
                        print('Ignored')
                        continue
                    if not self._cfg.DRY_RUN:
                        self.github_org.invite_user(user=user)
                        print('OK')
                    else:
                        print('Dry run')
                except GithubException as exc:
                    print(f'FAIL: {exc.data["errors"][0]["message"]}')


def _test():
    """Test and debug"""
    Config(cli_args=['DRY_RUN=True'])
    dev_emails = get_dev_emails()
    print('dev_emails:', dev_emails)

    gh_api = GithubOrgApi()
    gh_api.get_org_emails()


if __name__ == '__main__':
    _test()
