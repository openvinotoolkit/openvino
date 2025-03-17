# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Gets info about users and groups via LDAP
"""

# pylint: disable=fixme,no-member

import sys
from enum import Enum
from pathlib import Path

from ldap3 import Server, Connection, ALL, SUBTREE

sys.path.append(str(Path(__file__).resolve().parents[1]))
from github_org_control.configs import Config


class LdapApiException(Exception):
    """Base LDAP API exception"""


class InfoLevel(Enum):
    """Constants for printing user info from LDAP"""

    PDL = "PDL"  # Public Distribution List (group of e-mail addresses)
    FULL = "Full"


def print_user_info(info, info_level=None):
    """Pretty-print of a user info data structure (dict). info_level is the InfoLevel Enum"""
    if not info or not info.get("mail"):
        raise LdapApiException("ERROR: No info or absent mail")

    def get_membership():
        if info_level == InfoLevel.PDL:
            membership_info = "  PDLs:"
        elif info_level == InfoLevel.FULL:
            membership_info = "  memberOf :"
        else:
            return ""
        # Grouping groups by purpose
        if info_level == InfoLevel.PDL:
            sort_key = lambda i: i.split(",", 1)[0].lower()
        else:
            sort_key = lambda i: i.split(",", 1)[1] + i.split(",", 1)[0].lower()
        for item in sorted(info["memberOf"], key=sort_key):
            if info_level == InfoLevel.PDL and "OU=Delegated" not in item:
                continue
            membership_info += f"\n    {item}"
        return membership_info

    try:
        text_info = (
            f'\n{info["cn"]} <{info["mail"]}>; {info["sAMAccountName"]}; {info["employeeID"]}'
            f'\n  Org group: {info["intelSuperGroupDescr"]} ({info["intelSuperGroupShortName"]}) /'
            f' {info["intelGroupDescr"]} ({info["intelGroupShortName"]}) /'
            f' {info["intelDivisionDescr"]} ({info["intelDivisionShortName"]}) /'
            f' {info["intelOrgUnitDescr"]}'
            f'\n  Manager: {info.get("manager")}'
            f'\n  Location: {info["intelRegionCode"]} / {info["co"]} / {info["intelSiteCode"]} /'
            f' {info["intelBldgCode"]} ({info.get("intelSiteName")}) /'
            f' {info["physicalDeliveryOfficeName"]}'
            f'\n  Other: {info["employeeType"]} | {info["intelExportCountryGroup"]} |'
            f' {info["whenCreated"]} | {info["intelCostCenterDescr"]} | {info["jobDescription"]}'
        )
    except Exception as exc:
        raise LdapApiException(
            f'ERROR: Failed to get info about "{info["mail"]}". '
            f"Exception occurred:\n{repr(exc)}"
        ) from exc
    print(text_info)

    membership = get_membership()
    if info_level == InfoLevel.PDL and membership:
        print(membership)
    elif info_level == InfoLevel.FULL:
        for key in sorted(info):
            if isinstance(info[key], list):
                if key == "memberOf":
                    print(membership)
                else:
                    print(f"  {key} :")
                    for item in info[key]:
                        print("   ", item)
            else:
                print(f"  {key} : {info[key]}")


class LdapApi:
    """LDAP API for getting user info and emails"""

    _binary_blobs = ["thumbnailPhoto", "msExchUMSpokenName", "msExchBlockedSendersHash"]
    _check_existing = [
        "intelExportCountryGroup",
        "physicalDeliveryOfficeName",
        "intelSuperGroupShortName",
        "intelGroupShortName",
        "intelDivisionShortName",
    ]

    null = "<null>"

    def __init__(self):
        self._cfg = Config()
        self.server = Server(self._cfg.LDAP_SERVER, get_info=ALL)
        self.connection = Connection(
            self.server, user=self._cfg.LDAP_USER, password=self._cfg.LDAP_PASSWORD, auto_bind=True
        )
        self.connection.bind()

    def get_user_emails(self, groups=None):
        """Gets emails of LDAP groups and sub-groups"""
        print("\nGet emails from LDAP groups:")
        processed_ldap_members = {}

        def process_group_members(member, parent_group):
            if member in processed_ldap_members:
                processed_ldap_members[member]["parent_groups"].append(parent_group)
                print(
                    "\nWARNING: Ignore LDAP member to avoid duplication and recursive cycling "
                    f"of PDLs: {member}\n    "
                    f'email: {processed_ldap_members[member].get("email")}\n    parent_groups:'
                )
                for group in processed_ldap_members[member].get("parent_groups", []):
                    print(7 * " ", group)

                return
            processed_ldap_members[member] = {"email": None, "parent_groups": [parent_group]}

            # AD moves terminated users to the boneyard OU in case the user returns,
            # so it can be reactivated with little effort.
            # After 30 days it is removed and the unix personality becomes unlinked.
            if "OU=Boneyard" in member:
                return
            self.connection.search(
                member, r"(objectClass=*)", SUBTREE, attributes=["cn", "member", "mail"]
            )

            # print(self.connection.entries)
            if not self.connection.response:
                raise LdapApiException(f"ERROR: empty response. LDAP member: {member}")

            # Check that the member is worker.
            # The response can contain several items, but the first item is valid only
            if "OU=Workers" in member:
                if self.connection.response[0]["attributes"]["mail"]:
                    processed_ldap_members[member]["email"] = self.connection.response[0][
                        "attributes"
                    ]["mail"].lower()
                    return
                raise LdapApiException(
                    f"ERROR: no mail. LDAP worker: {member}\n" f"{self.connection.entries}"
                )

            if len(self.connection.response) > 1:
                raise LdapApiException(
                    f"ERROR: multiple responses for {member}: "
                    f"{len(self.connection.response)}\n"
                    f"{self.connection.entries}"
                )

            if self.connection.response[0]["attributes"]["member"]:
                for group_member in self.connection.response[0]["attributes"]["member"]:
                    process_group_members(group_member, member)
            else:
                print(f"\nERROR: no members in LDAP group: {member}\n{self.connection.entries}")

        for group in groups or self._cfg.LDAP_PDLs:
            print("\nProcess ROOT LDAP group:", group)
            process_group_members(group, "ROOT")
        return {
            member.get("email") for member in processed_ldap_members.values() if member.get("email")
        }

    def _get_user_info(self, query):
        """Gets user info from LDAP as dict matching key and values pairs from query"""
        query_filter = "".join(f"({key}={value})" for key, value in query.items())

        for domain in self._cfg.LDAP_DOMAINS:
            search_base = f"OU=Workers,DC={domain},DC=corp,DC=intel,DC=com"
            self.connection.search(
                search_base,
                f"(&(objectcategory=person)(objectclass=user)(intelflags=1){query_filter})",
                SUBTREE,
                attributes=["*"],
            )

            if self.connection.response:
                if len(self.connection.response) > 1:
                    raise LdapApiException(
                        f"ERROR: multiple responses for {query_filter}: "
                        f"{len(self.connection.response)}\n"
                        f"{self.connection.entries}"
                    )
                info = self.connection.response[0]["attributes"]

                # remove long binary blobs
                for blob in LdapApi._binary_blobs:
                    info[blob] = b""
                for key in LdapApi._check_existing:
                    if not info.get(key):
                        info[key] = LdapApi.null
                return info
        return {}

    def get_user_info_by_idsid(self, idsid):
        """Gets user info from LDAP as dict using account name for searching"""
        return self._get_user_info({"sAMAccountName": idsid})

    def get_user_info_by_name(self, name):
        """Gets user info from LDAP as dict using common name for searching"""
        return self._get_user_info({"cn": name})

    def get_user_info_by_email(self, email):
        """Gets user info from LDAP as dict using emails for searching"""
        return self._get_user_info({"mail": email})

    def get_absent_emails(self, emails):
        """Checks users by email in LDAP and returns absent emails"""
        absent_emails = set()
        for email in emails:
            if not self.get_user_info_by_email(email):
                absent_emails.add(email)
        return absent_emails


def _test():
    """Test and debug"""
    ldap = LdapApi()

    emails = ldap.get_user_emails()
    print(f'\nLDAP emails count: {len(emails)}\n{"; ".join(emails)}')

    emails = ["foo@intel.com"]

    for email in emails:
        info = ldap.get_user_info_by_email(email)
        if info:
            print_user_info(info, InfoLevel.PDL)
        else:
            print(f"\n{email} - not found")


if __name__ == "__main__":
    _test()
