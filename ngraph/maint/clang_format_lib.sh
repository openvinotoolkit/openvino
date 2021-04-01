#!/bin/bash

# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

#===================================================================================================
# Provides Bash functions for dealing with clang-format.
#===================================================================================================

declare _intelnervana_clang_format_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_clang_format_lib_SCRIPT_NAME}") && pwd )"

source "${_maint_SCRIPT_DIR}/bash_lib.sh"

clang_format_lib_verify_version() {
    if (( $# != 2 )); then
        bash_lib_print_error "Usage: ${FUNCNAME[0]} <clang-format-prog-pathname> <required-version-number>"
        return 1
    fi

    local PROGNAME="${1}"
    local REQUIRED_VERSION_X_Y="${2}"

    if ! [[ "${REQUIRED_VERSION_X_Y}" =~ ^[0-9]+.[0-9]+$ ]]; then
        bash_lib_print_error "${FUNCNAME[0]}: required-version-number must have the form (number).(number)."
        return 1
    fi

    if ! [[ -f "${PROGNAME}" ]]; then
        bash_lib_print_error "Unable to find clang-format program named '${PROGNAME}'"
        return 1
    fi

    local VERSION_LINE
    if ! VERSION_LINE=$("${PROGNAME}" --version); then
        bash_lib_print_error "Failed invocation of command '${PROGNAME} --version'"
        return 1
    fi

    local SED_FLAGS
    if [[ "$(uname)" == 'Darwin' ]]; then
        SED_FLAGS='-En'
    else
        SED_FLAGS='-rn'
    fi

    local VERSION_X_Y
    if ! VERSION_X_Y=$(echo "${VERSION_LINE}" | sed ${SED_FLAGS} 's/^clang-format version ([0-9]+.[0-9]+).*$/\1/p')
    then
        bash_lib_print_error "Failed invocation of sed."
        return 1
    fi

    if [[ "${REQUIRED_VERSION_X_Y}" != "${VERSION_X_Y}" ]]; then
        bash_lib_print_error \
            "Program '${PROGNAME}' reports version number '${VERSION_X_Y}'" \
            "but we require '${REQUIRED_VERSION_X_Y}'"
        return 1
    fi
}
