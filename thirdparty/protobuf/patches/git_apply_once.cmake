# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Idempotently apply a git patch to the tree in the current working directory.
# Used as a FetchContent PATCH_COMMAND; FetchContent runs it with the working
# directory set to the dependency's populated source tree.
#
# Usage:
#   cmake -DPATCH=<absolute-path-to-.patch> -P git_apply_once.cmake
#
# If the patch already applies in reverse, it is considered applied and the
# step is skipped, so re-running configure/populate does not fail.

if(NOT DEFINED PATCH OR PATCH STREQUAL "")
    message(FATAL_ERROR "git_apply_once.cmake: PATCH argument is required")
endif()

if(NOT EXISTS "${PATCH}")
    message(FATAL_ERROR "git_apply_once.cmake: patch file does not exist: ${PATCH}")
endif()

find_package(Git QUIET REQUIRED)

# Reverse-check succeeds only if the patch is already applied -> nothing to do.
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false apply
        --reverse --check --ignore-whitespace --ignore-space-change "${PATCH}"
    RESULT_VARIABLE _reverse_rc
    OUTPUT_QUIET
    ERROR_QUIET)

if(_reverse_rc EQUAL 0)
    message(STATUS "git_apply_once: patch already applied: ${PATCH}")
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false apply
        --ignore-whitespace --ignore-space-change "${PATCH}"
    RESULT_VARIABLE _apply_rc
    ERROR_VARIABLE _apply_err)

if(NOT _apply_rc EQUAL 0)
    message(FATAL_ERROR "git_apply_once: failed to apply ${PATCH}:\n${_apply_err}")
endif()

message(STATUS "git_apply_once: applied patch: ${PATCH}")
