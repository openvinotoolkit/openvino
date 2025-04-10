#!/bin/bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Redirects git submodules to gitee mirrors and updates these recursively.
#
# To revert the changed URLs use 'git submodule deinit .'
#

# -----------------------------------------------------------------------------
# Common bash

if [[ -n ${DEBUG_SHELL} ]]
then
  set -x # Activate the expand mode if DEBUG is anything but empty.
fi

set -o errexit # Exit if command failed.
set -o pipefail # Exit if pipe failed.
set -o nounset # Exit if variable not set.

die() {
    echo "${1:-"Unknown Error"}" 1>&2
    exit 1
}

check_git_version() {
    currentver="$(git --version 2>&1 |awk 'NR==1{gsub(/"/,"");print $3}')"
    requiredver="2.11.0"
    if [ "$(printf '%s\n' "$requiredver" "$currentver" | sort -V | head -n1)" = "$requiredver" ]; then
        # Greater than or equal to 2.11.0
        true
    else
        # Less than 2.11.0
        false
    fi
}

# -----------------------------------------------------------------------------

ERR_CANNOT_UPDATE=13
GITEE_GROUP_NAME="openvinotoolkit-prc"

REPO_DIR=${1:-"${PWD}"}
REPO_DIR=$(cd "${REPO_DIR}" && pwd -P)

SCRIPT_SH=$(cd "$(dirname "${0}")" && pwd -P)/$(basename "${0}")

[ -d "${REPO_DIR}" ] || die "${REPO_DIR} is not directory!"
[ -f "${SCRIPT_SH}" ] || die "${SCRIPT_SH} does not exist!"

pushd "${REPO_DIR}" >/dev/null

# Step 0: Check if .gitmodules file exsit, otherwise no submodule update for this repo
[ -f ".gitmodules" ] || exit 0

# Step 1: Init git submodule
git submodule init

# Step 2: Replacing each submodule URL of the current repository to the mirror repos in gitee
for LINE in $(git config -f .gitmodules --list | grep "\.url=../../[^.]\|\.url=https://github.com/[^.]\|\.url=https://git.eclipse.org/[^.]")
do
    SUBPATH="${LINE//.url*/}"
    LOCATION=$(echo "${LINE}" | sed 's/.*\///' | sed 's/.git//g' | sed 's/.*\.//')
    if [ "$LOCATION" = "unity" ]; then
	    LOCATION="Unity"
    fi
    SUBURL="https://gitee.com/$GITEE_GROUP_NAME/$LOCATION"
    git config submodule."${SUBPATH}".url "${SUBURL}"
done

# Step 3: Getting submodules of the current repository from gitee mirrors
if check_git_version; then
    git submodule update --progress || exit $ERR_CANNOT_UPDATE
else
    git submodule update || exit $ERR_CANNOT_UPDATE
fi

# Step 4: Replacing URLs for each sub-submodule. The script runs recursively
git submodule foreach "${SCRIPT_SH}"

popd >/dev/null
