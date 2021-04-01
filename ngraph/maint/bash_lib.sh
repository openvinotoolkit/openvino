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
# A library of general-purpose Bash functions
#===================================================================================================

declare _intelnervana_bash_lib_SCRIPT_NAME="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
declare _maint_SCRIPT_DIR="$( cd $(dirname "${_intelnervana_bash_lib_SCRIPT_NAME}") && pwd )"
declare _intelnervana_bash_lib_IS_LOADED=1

bash_lib_get_my_BASH_LINENO() {
    echo "${BASH_LINENO[${#BASH_LINENO[@]} -1 ]}"
}

bash_lib_get_callers_BASH_LINENO() {
    echo "${BASH_LINENO[${#BASH_LINENO[@]} - 2]}"
}

bash_lib_get_my_BASH_SOURCE() {
    echo "${BASH_SOURCE[${#BASH_SOURCE[@]} ]}"
}

bash_lib_get_callers_BASH_SOURCE() {
    echo "${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"
}

bash_lib_status() {
    local CONTEXT_STRING="$(basename $(bash_lib_get_callers_BASH_SOURCE))"
    local TEXT_LINE
    local IS_FIRST_LINE=1

    for TEXT_LINE in "${@}"; do
        if (( IS_FIRST_LINE == 1 )); then
            IS_FIRST_LINE=0
            printf "%s STATUS: " "${CONTEXT_STRING}" >&2
        else
            printf "    " >&2
        fi

        printf "%s\n" "${TEXT_LINE}" >&2
    done
}

bash_lib_print_error() {
    local CONTEXT_STRING="$(basename $(bash_lib_get_callers_BASH_SOURCE)):$(bash_lib_get_callers_BASH_LINENO)"
    local TEXT_LINE
    local IS_FIRST_LINE=1

    for TEXT_LINE in "${@}"; do
        if (( IS_FIRST_LINE == 1 )); then
            IS_FIRST_LINE=0
            printf "%s ERROR: " "${CONTEXT_STRING}" >&2
        else
            printf "    " >&2
        fi

        printf "%s\n" "${TEXT_LINE}" >&2
    done
}

bash_lib_die() {
    bash_lib_print_error $@
    exit 1
}

bash_lib_am_sudo_or_root() {
    [ "$EUID" -eq 0 ]
}

if bash_lib_am_sudo_or_root; then
    bash_lib_MAYBE_SUDO=''
else
    bash_lib_MAYBE_SUDO='sudo --set-home'
fi
