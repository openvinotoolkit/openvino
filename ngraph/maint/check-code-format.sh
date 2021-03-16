#!/bin/bash
set -e
set -u

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

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.

declare CLANG_FORMAT_BASENAME="clang-format-9"
declare REQUIRED_CLANG_FORMAT_VERSION=9.0

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

clang_format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}"
echo "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

declare -a FAILED_FILES=()
declare NUM_FILES_CHECKED=0

pushd "${THIS_SCRIPT_DIR}/.."

declare PYBIND_WRAPPER="python/pyngraph"

declare ROOT_SUBDIR
for ROOT_SUBDIR in core test frontend python/src/pyngraph; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
        echo "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found."
    else
        echo "About to format C/C++ code in directory tree '$(pwd)/${ROOT_SUBDIR}' ..."
        declare SRC_FILE
        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        for SRC_FILE in $(find "${ROOT_SUBDIR}" -type f -and \( -name '*.cpp' -or -name '*.hpp' \) ); do
            if "${CLANG_FORMAT_PROG}" -style=file -output-replacements-xml "${SRC_FILE}" | grep -c "<replacement " >/dev/null; then
                FAILED_FILES+=( "${SRC_FILE}" )
            fi
            NUM_FILES_CHECKED=$((NUM_FILES_CHECKED+1))
        done
    fi
done

popd

if [[ ${#FAILED_FILES[@]} -eq 0 ]]; then
    echo "All ${NUM_FILES_CHECKED}  C/C++ files pass the code-format check."
else
    echo "${#FAILED_FILES[@]} of ${NUM_FILES_CHECKED} source files failed the code-format check:"
    declare FAILED_SRC_FILE
    for FAILED_SRC_FILE in ${FAILED_FILES[@]}; do
        echo "    ${FAILED_SRC_FILE}"
    done
    exit 1
fi
