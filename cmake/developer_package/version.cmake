# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (branchName VAR)
    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()
    execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${repo_root}
            OUTPUT_VARIABLE GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set (${VAR} ${GIT_BRANCH} PARENT_SCOPE)
endfunction()

function (commitHash VAR)
    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()
    execute_process(
            COMMAND git rev-parse HEAD
            WORKING_DIRECTORY ${repo_root}
            OUTPUT_VARIABLE GIT_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set (${VAR} ${GIT_COMMIT_HASH} PARENT_SCOPE)
endfunction()

macro(ie_parse_ci_build_number)
    if(CI_BUILD_NUMBER MATCHES "^([0-9]+)\.([0-9]+)\.([0-9]+)\-.*")
        set(IE_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(IE_VERSION_MINOR ${CMAKE_MATCH_2})
        set(IE_VERSION_PATCH ${CMAKE_MATCH_3})
    endif()

    # TODO:
    # parse static version from ie_version.hpp
    # and compare with information from CI_BUILD_NUMBER

    set(IE_VERSION "${IE_VERSION_MAJOR}.${IE_VERSION_MINOR}.${IE_VERSION_PATCH}")
endmacro()


if (DEFINED ENV{CI_BUILD_NUMBER})
    set(CI_BUILD_NUMBER $ENV{CI_BUILD_NUMBER})
else()
    branchName(GIT_BRANCH)
    commitHash(GIT_COMMIT_HASH)

    set(custom_build "custom_${GIT_BRANCH}_${GIT_COMMIT_HASH}")
    set(CI_BUILD_NUMBER "${custom_build}")
endif()

function (addVersionDefines FILE)
    foreach (VAR ${ARGN})
        if (DEFINED ${VAR} AND NOT "${${VAR}}" STREQUAL "")
            set_property(
                SOURCE ${FILE}
                APPEND
                PROPERTY COMPILE_DEFINITIONS
                ${VAR}="${${VAR}}")
        endif()
    endforeach()
endfunction()
