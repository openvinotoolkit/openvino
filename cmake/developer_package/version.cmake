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
        set(has_ci_version ON)
    else()
        set(IE_VERSION_MAJOR 0)
        set(IE_VERSION_MINOR 0)
        set(IE_VERSION_PATCH 0)
    endif()

    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()

    if(DEFINED IEDevScripts_DIR AND DEFINED IE_MAIN_SOURCE_DIR AND NOT DEFINED custom_build)
        set(ie_version_hpp "${IE_MAIN_SOURCE_DIR}/include/ie_version.hpp")
        if(NOT EXISTS ${ie_version_hpp})
            message(FATAL_ERROR "File ie_version.hpp with IE_VERSION definitions is not found")
        endif()

        file(STRINGS "${ie_version_hpp}" IE_VERSION_PARTS REGEX "#define IE_VERSION_[A-Z]+[ ]+" )

        string(REGEX REPLACE ".+IE_VERSION_MAJOR[ ]+([0-9]+).*" "\\1"
               IE_VERSION_MAJOR_HPP "${IE_VERSION_PARTS}")
        string(REGEX REPLACE ".+IE_VERSION_MINOR[ ]+([0-9]+).*" "\\1"
               IE_VERSION_MINOR_HPP "${IE_VERSION_PARTS}")
        string(REGEX REPLACE ".+IE_VERSION_PATCH[ ]+([0-9]+).*" "\\1"
               IE_VERSION_PATCH_HPP "${IE_VERSION_PARTS}")

        foreach(var IE_VERSION_MAJOR IE_VERSION_MINOR IE_VERSION_PATCH)
            if(DEFINED ${var} AND NOT ${var} EQUAL ${var}_HPP)
                message(FATAL_ERROR "${var} parsed from CI_BUILD_NUMBER (${${var}}) \
                    and from ie_version.hpp (${${var}_HPP}) are different")
            else()
                # CI_BUILD_NUMBER is not defined well, take info from ie_verison.hpp as a baseline
                set(${var} ${${var}_HPP})
            endif()
        endforeach()
    elseif(has_ci_version)
        message(WARNING "IE_MAIN_SOURCE_DIR is not defined. No way to compare versions")
    else()
        message(WARNING "No way to detect OpenVINO version. Supposing 0.0.0.0")
    endif()

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

# provides Inference Engine version
# 1. If CI_BUILD_NUMBER is defined, parses this information
# 2. Otherwise, parses ie_version.hpp
ie_parse_ci_build_number()

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
