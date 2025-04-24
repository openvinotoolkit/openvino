# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
GitHub API for controlling organization
"""

# pylint: disable=fixme,no-member

import re
import sys
import time
import typing
from pathlib import Path

from github import Github, GithubException, RateLimitExceededException, IncompletableObject
from github.PaginatedList import PaginatedList

sys.path.append(str(Path(__file__).resolve().parents[1]))
from github_org_control.configs import Config


class GithubApiException(Exception):
    """Base GitHub API exception"""


def is_valid_user(user):
    """Checks that user is valid github.Github object"""
    try:
        return user and user.login
    except IncompletableObject:
        return False


def is_user_ignored(user):
    """Checks that user should be ignored"""
    if is_valid_user(user) and user.login.lower() not in Config().IGNORE_LOGINS:
        return False
    return True


def is_valid_name(name):
    """Checks that GitHub user's name is valid"""
    return name and len(name) >= 3 and " " in name


def is_intel_email(email):
    """Checks that email is valid Intel email"""
    return email and len(email) > 10 and " " not in email and email.lower().endswith("@intel.com")


def is_intel_company(company):
    """Checks that company contains intel"""
    return company and "intel" in company.lower()


def is_valid_intel_user(user):
    """Checks that user is valid GitHub and Intel user"""
    try:
        return is_valid_user(user) and is_valid_name(user.name) and is_intel_email(user.email)
    except IncompletableObject:
        return False


def print_users(users):
    """Print list of users in different formats: list, set, PaginatedList"""
    if isinstance(users, (list, set, PaginatedList)):
        users_count = users.totalCount if isinstance(users, PaginatedList) else len(users)
        print(f"GitHub users {users_count} (login - name - company - email - valid):")
    else:
        users = [users]
    for user in users:
        if not is_valid_user(user):
            print("WRONG GitHub user: ???")
            continue

        try:
            name = user.name
        except IncompletableObject:
            name = "???"

        try:
            company = user.company
        except IncompletableObject:
            company = "???"

        try:
            email = user.email
        except IncompletableObject:
            email = "???"

        valid_check = "OK" if is_valid_intel_user(user) else "FIX"
        if not is_intel_email(email):
            valid_check += " email"
        if not is_valid_name(name):
            valid_check += " name"
        print(f'{user.login} - "{name}" - "{company}" - {email} - {valid_check}')


def get_dev_emails():
    """
    Read a file with developer emails. Supported email formats
    first_name.last_name@intel.com
    Import from Outlook: Last_name, First_name <first_name.last_name@intel.com>
    """
    re_email = re.compile(r".+<(.+)>")
    emails = set()
    cfg = Config()
    with open(cfg.properties["EMAILS_FILE_PATH"]) as file_obj:
        for line in file_obj:
            line = line.strip().lower()
            if not line or line.startswith("#"):
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
        self.repo = self.github.get_repo(f"{self._cfg.GITHUB_ORGANIZATION}/{self._cfg.GITHUB_REPO}")
        self.github_users_by_email = {}
        self.org_members_by_login = {}
        self.members_to_remove = set()
        self.members_to_fix_name = set()

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

        print(f"\nOrg members {org_members.totalCount} (login - name - company - email - valid):")
        for org_member in org_members:
            self.org_members_by_login[org_member.login.lower()] = org_member
            print_users(org_member)
            if is_intel_email(org_member.email):
                email = org_member.email.lower()
                org_emails.add(email)
                self.github_users_by_email[email] = org_member
                if not is_valid_name(org_member.name):
                    self.members_to_fix_name.add(org_member)
            else:
                self.members_to_remove.add(org_member)

        print("\nOrg members - no Intel emails:")
        print_users(self.members_to_remove)

        print("\nOrg members - no real name:")
        print_users(self.members_to_fix_name)
        print(
            "\nOrg member emails - no real name:",
            "; ".join([member.email.lower() for member in self.members_to_fix_name]),
        )

        return org_emails

    def get_org_invitation_emails(self):
        """Gets GitHub organization teams prints info"""
        org_invitations = self.github_org.invitations()
        org_invitation_emails = set()

        print(
            f"\nOrg invitations {org_invitations.totalCount} "
            "(login - name - company - email - valid):"
        )
        for org_invitation in org_invitations:
            print_users(org_invitation)
            if is_user_ignored(org_invitation):
                continue
            if is_intel_email(org_invitation.email):
                org_invitation_emails.add(org_invitation.email.lower())
            else:
                print("Strange org invitation:", org_invitation)

        print(
            f"\nOrg invitation emails {len(org_invitation_emails)}:",
            "; ".join(org_invitation_emails),
        )
        return org_invitation_emails

    def get_org_teams(self):
        """Gets GitHub organization teams prints info"""
        teams = []
        org_teams = self.github_org.get_teams()
        print("\nOrg teams count:", org_teams.totalCount)
        for team in org_teams:
            teams.append(team.name)
            print(f"\nTeam: {team.name} - parent: {team.parent}")

            repos = team.get_repos()
            print("Repos:")
            for repo in repos:
                print(f"    {repo.name} -", team.get_repo_permission(repo))

            team_maintainers = team.get_members(role="maintainer")
            team_maintainer_logins = set()
            for maintainer in team_maintainers:
                team_maintainer_logins.add(maintainer.login)
            team_members = team.get_members(role="member")
            team_member_logins = set()
            for member in team_members:
                team_member_logins.add(member.login)
            members = team.get_members(role="all")
            member_emails = []
            print("Members (role - login - name - company - email - valid):")
            for user in members:
                if user.login in team_maintainer_logins:
                    print("    Maintainer - ", end="")
                elif user.login in team_member_logins:
                    print("    Member - ", end="")
                else:
                    # It is not possible to check child teams members
                    print("    ??? - ", end="")
                print_users(user)
                if is_intel_email(user.email) and not is_user_ignored(user):
                    member_emails.append(user.email.lower())
            print(f"Intel emails {len(member_emails)}:", "; ".join(member_emails))
        return teams

    def get_github_user_by_email(self, email):
        """Gets GitHub user by email"""
        if email in self.github_users_by_email:
            return self.github_users_by_email.get(email)

        def search_users():
            paginated_users = self.github.search_users(f"{email} in:email")
            # Minimize the GitHub Rate Limit
            users = []
            for user in paginated_users:
                users.append(user)
            if len(users) == 1:
                return users[0]
            if len(users) == 0:
                return None
            raise GithubApiException(
                f"ERROR: Found {len(users)} GitHub accounts with the same email {email}"
            )

        try:
            user = search_users()
        except RateLimitExceededException:
            print("WARNING: RateLimitExceededException")
            time.sleep(30)
            user = search_users()
        self.github_users_by_email[email] = user

        return user

    def get_valid_github_users(self, emails):
        """Gets valid GitHub users by email and prints status"""
        valid_users = set()
        wrong_emails = set()
        no_account_emails = set()
        no_account_names = set()
        print(f"\nGitHub users from {len(emails)} invite emails (email - status):")
        for email in emails:
            if not is_intel_email(email):
                print(f"{email} - Non Intel email")
                wrong_emails.add(email)
                continue

            # You can make up to 30 requests per minute; https://developer.github.com/v3/search/
            time.sleep(2)
            user = self.get_github_user_by_email(email)

            if not user:
                print(f"{email} - No valid GitHub account")
                no_account_emails.add(email)
                continue

            if user.email and user.email.lower() == email:
                if is_valid_name(user.name):
                    print(f"{email} - OK")
                    valid_users.add(user)
                else:
                    print(f"{email} - No valid name in GitHub account: ", end="")
                    print_users(user)
                    no_account_names.add(email)
            else:
                print(f"{email} - Non public or wrong email in GitHub account: ", end="")
                print_users(user)
                no_account_emails.add(email)

        print("\nValid users:")
        print_users(valid_users)

        print(f"\nWrong emails {len(wrong_emails)}:", "; ".join(wrong_emails))

        print(
            f"\nIntel emails - No valid GitHub account {len(no_account_emails)}:",
            "; ".join(no_account_emails),
        )

        print(
            f"\nIntel emails - No valid name in GitHub account {len(no_account_names)}:",
            "; ".join(no_account_names),
        )
        return valid_users

    def invite_users(self, users):
        """Invites users to GitHub organization and prints status"""
        if not isinstance(users, typing.Iterable):
            users = [users]
        print(f"\nInvite {len(users)} users:")

        for user in users:
            if isinstance(user, str):
                print(f"Email: {user}")
                self.github_org.invite_user(email=user)
            else:
                print(f'{user.login} - "{user.name}" - {user.email} - ', end="")
                try:
                    if is_user_ignored(user):
                        print("Ignored")
                        continue
                    if self._cfg.DRY_RUN:
                        print("Dry run")
                        continue
                    self.github_org.invite_user(user=user)
                    print("OK")
                except GithubException as exc:
                    print(f'FAIL: {exc.data["errors"][0]["message"]}')

    def remove_users(self, users):
        """Removes users from GitHub organization"""
        if not isinstance(users, typing.Iterable):
            users = [users]
        print(f"\nRemove {len(users)} users:")

        dry_run = self._cfg.DRY_RUN
        if not dry_run and len(users) > self._cfg.MAX_MEMBERS_TO_REMOVE:
            print(
                "WARNING: Review is required for removing members more than "
                f"{self._cfg.MAX_MEMBERS_TO_REMOVE}"
            )
            # TODO: Add notification
            dry_run = True

        for user in users:
            member = self.get_github_user_by_email(user) if isinstance(user, str) else user
            print(f'{member.login} - "{member.name}" - {member.email} - ', end="")
            try:
                if is_user_ignored(member):
                    print("Ignored")
                    continue
                if dry_run:
                    print("Dry run")
                    continue
                self.github_org.remove_from_membership(member)
                print("OK")
            except GithubException as exc:
                print(f'FAIL: {exc.data["errors"][0]["message"]}')


def _test():
    """Test and debug"""
    Config(cli_args=["DRY_RUN=True"])
    dev_emails = get_dev_emails()
    print("dev_emails:", dev_emails)

    gh_api = GithubOrgApi()
    gh_api.get_org_emails()


if __name__ == "__main__":
    _test()
