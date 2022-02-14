# Copyright (C) 2018-2022 Intel Corporation
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

macro(ov_parse_ci_build_number)
    set(OpenVINO_VERSION_BUILD 000)
    set(IE_VERSION_BUILD ${OpenVINO_VERSION_BUILD})

    if(CI_BUILD_NUMBER MATCHES "^([0-9]+)\.([0-9]+)\.([0-9]+)\-([0-9]+)\-.*")
        set(OpenVINO_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(OpenVINO_VERSION_MINOR ${CMAKE_MATCH_2})
        set(OpenVINO_VERSION_PATCH ${CMAKE_MATCH_3})
        set(OpenVINO_VERSION_BUILD ${CMAKE_MATCH_4})
    endif()

    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()

    macro(ov_get_hpp_version)
        if(NOT DEFINED OpenVINO_SOURCE_DIR)
            return()
        endif()

        set(ie_version_hpp "${OpenVINO_SOURCE_DIR}/src/inference/include/ie/ie_version.hpp")
        if(NOT EXISTS ${ie_version_hpp})
            message(FATAL_ERROR "File ie_version.hpp with IE_VERSION definitions is not found")
        endif()

        set(ov_version_hpp "${OpenVINO_SOURCE_DIR}/src/core/include/openvino/core/version.hpp")
        if(NOT EXISTS ${ov_version_hpp})
            message(FATAL_ERROR "File openvino/core/version.hpp with OPENVINO_VERSION definitions is not found")
        endif()

        file(STRINGS "${ie_version_hpp}" IE_VERSION_PARTS REGEX "#define IE_VERSION_[A-Z]+[ ]+" )
        file(STRINGS "${ov_version_hpp}" OV_VERSION_PARTS REGEX "#define OPENVINO_VERSION_[A-Z]+[ ]+" )

        foreach(suffix MAJOR MINOR PATCH)
            set(ie_version_name "IE_VERSION_${suffix}")
            set(ov_version_name "OpenVINO_VERSION_${suffix}")
            set(ov_version_name_hpp "OPENVINO_VERSION_${suffix}")

            string(REGEX REPLACE ".+${ie_version_name}[ ]+([0-9]+).*" "\\1"
                    ${ie_version_name}_HPP "${IE_VERSION_PARTS}")
            string(REGEX REPLACE ".+${ov_version_name_hpp}[ ]+([0-9]+).*" "\\1"
                    ${ov_version_name}_HPP "${OV_VERSION_PARTS}")

            if(NOT ${ie_version_name}_HPP EQUAL ${ov_version_name}_HPP)
                message(FATAL_ERROR "${ov_version_name} (${${ov_version_name}_HPP})"
                                    " and ${ie_version_name} (${${ie_version_name}_HPP}) are not equal")
            endif()
        endforeach()

        set(ov_hpp_version_is_found ON)
    endmacro()

    # detect OpenVINO version via openvino/core/version.hpp and ie_version.hpp
    ov_get_hpp_version()

    if(ov_hpp_version_is_found)
        foreach(var OpenVINO_VERSION_MAJOR OpenVINO_VERSION_MINOR OpenVINO_VERSION_PATCH)
            if(DEFINED ${var} AND NOT ${var} EQUAL ${var}_HPP)
                message(FATAL_ERROR "${var} parsed from CI_BUILD_NUMBER (${${var}}) \
                    and from openvino/core/version.hpp (${${var}_HPP}) are different")
            else()
                # CI_BUILD_NUMBER is not defined well, take info from openvino/core/version.hpp as a baseline
                set(${var} ${${var}_HPP})
            endif()
        endforeach()
    endif()

    set(OpenVINO_VERSION "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}")
    message(STATUS "OpenVINO version is ${OpenVINO_VERSION} (Build ${OpenVINO_VERSION_BUILD})")
endmacro()

if (DEFINED ENV{CI_BUILD_NUMBER})
    set(CI_BUILD_NUMBER $ENV{CI_BUILD_NUMBER})
else()
    branchName(GIT_BRANCH)
    commitHash(GIT_COMMIT_HASH)

    set(custom_build "custom_${GIT_BRANCH}_${GIT_COMMIT_HASH}")
    set(CI_BUILD_NUMBER "${custom_build}")
endif()

# provides OpenVINO version
# 1. If CI_BUILD_NUMBER is defined, parses this information
# 2. Otherwise, parses openvino/core/version.hpp
ov_parse_ci_build_number()

macro (addVersionDefines FILE)
    set(__version_file ${FILE})
    if(NOT IS_ABSOLUTE ${__version_file})
        set(__version_file "${CMAKE_CURRENT_SOURCE_DIR}/${__version_file}")
    endif()
    if(NOT EXISTS ${__version_file})
        message(FATAL_ERROR "${FILE} does not exists in current source directory")
    endif()
    foreach (VAR ${ARGN})
        if (DEFINED ${VAR} AND NOT "${${VAR}}" STREQUAL "")
            set_property(
                SOURCE ${__version_file}
                APPEND
                PROPERTY COMPILE_DEFINITIONS
                ${VAR}="${${VAR}}")
        endif()
    endforeach()
    unset(__version_file)
endmacro()
