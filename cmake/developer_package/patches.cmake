# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# ov_apply_patch(
#   PATCH <absolute-path-to-.patch-file>
#   [WORKING_DIRECTORY <dir>]   -- directory passed to 'git apply'; defaults to CMAKE_SOURCE_DIR
# )
#
# Applies a local patch file via 'git apply' during the CMake configure phase.
# Idempotent: uses 'git apply --reverse --check' to detect whether the patch is
# already applied and skips the apply step if so.
# Relies on GIT_FOUND / GIT_EXECUTABLE, which are set by find_package(Git) in version.cmake.
#
function(ov_apply_patch)
    cmake_parse_arguments(ARG "" "PATCH;WORKING_DIRECTORY" "" ${ARGN})

    if(NOT DEFINED ARG_PATCH OR ARG_PATCH STREQUAL "")
        message(FATAL_ERROR "ov_apply_patch: PATCH argument is required")
    endif()

    if(NOT EXISTS "${ARG_PATCH}")
        message(FATAL_ERROR "ov_apply_patch: patch file does not exist: ${ARG_PATCH}")
    endif()

    if(NOT DEFINED ARG_WORKING_DIRECTORY OR ARG_WORKING_DIRECTORY STREQUAL "")
        set(ARG_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endif()

    if(NOT IS_DIRECTORY "${ARG_WORKING_DIRECTORY}")
        message(FATAL_ERROR "ov_apply_patch: working directory does not exist: ${ARG_WORKING_DIRECTORY}")
    endif()

    # GIT_FOUND / GIT_EXECUTABLE are set by find_package(Git) in version.cmake,
    # but may be absent (e.g. Git not on PATH at configure time on Windows).
    # Re-run find_package locally as a fallback — it is cheap and cached by CMake.
    if(NOT GIT_FOUND)
        find_package(Git QUIET)
    endif()
    if(NOT GIT_FOUND)
        message(FATAL_ERROR
            "ov_apply_patch: Git is required to apply patches but was not found.\n"
            "  Patch: ${ARG_PATCH}")
    endif()

    get_filename_component(_patch_display "${ARG_PATCH}" NAME)

    # Check whether the patch is already applied by attempting a dry-run reverse.
    # Exit code 0  → reverse applies cleanly → patch IS already applied → skip.
    # Exit code ≠ 0 → reverse fails           → patch is NOT yet applied → apply it.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false apply
            --reverse --check
            --ignore-space-change
            --ignore-whitespace
            "${ARG_PATCH}"
        WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        RESULT_VARIABLE _ov_reverse_rc
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(_ov_reverse_rc EQUAL 0)
        return()
    endif()

    # Patch not yet applied — apply it now.
    # -c core.autocrlf=false: patch files on Windows may have CRLF line endings;
    # disabling autocrlf makes git apply agnostic of the patch file's line endings.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false apply
            --ignore-space-change
            --ignore-whitespace
            "${ARG_PATCH}"
        WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        RESULT_VARIABLE _ov_patch_rc
        ERROR_VARIABLE  _ov_patch_err
        OUTPUT_QUIET
    )

    if(NOT _ov_patch_rc EQUAL 0)
        message(FATAL_ERROR
            "ov_apply_patch: failed to apply '${_patch_display}':\n${_ov_patch_err}")
    endif()

    message(STATUS "Applied patch: ${_patch_display}")
endfunction()
