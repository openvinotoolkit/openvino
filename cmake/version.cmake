# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function (branchName VAR)
    execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${OpenVINO_MAIN_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set (${VAR} ${GIT_BRANCH} PARENT_SCOPE)
endfunction()

function (commitHash VAR)
    execute_process(
            COMMAND git rev-parse HEAD
            WORKING_DIRECTORY ${OpenVINO_MAIN_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set (${VAR} ${GIT_COMMIT_HASH} PARENT_SCOPE)
endfunction()

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
