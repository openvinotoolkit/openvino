# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(Git QUIET)

function (branchName VAR)
    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY ${repo_root}
                OUTPUT_VARIABLE GIT_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        set (${VAR} ${GIT_BRANCH} PARENT_SCOPE)
    endif()
endfunction()

function (commitHash VAR)
    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-parse --short=11 HEAD
                WORKING_DIRECTORY ${repo_root}
                OUTPUT_VARIABLE GIT_COMMIT_HASH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        set (${VAR} ${GIT_COMMIT_HASH} PARENT_SCOPE)
    endif()
endfunction()

function (commitNumber VAR)
    if(NOT DEFINED repo_root)
        message(FATAL_ERROR "repo_root is not defined")
    endif()
    if(GIT_FOUND)
        execute_process(
                COMMAND ${GIT_EXECUTABLE} rev-list --count --first-parent HEAD
                WORKING_DIRECTORY ${repo_root}
                OUTPUT_VARIABLE GIT_COMMIT_NUMBER
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        set (${VAR} ${GIT_COMMIT_NUMBER} PARENT_SCOPE)
    endif()
endfunction()

macro(ov_parse_ci_build_number)
    set(OpenVINO_VERSION_BUILD 000)

    if(CI_BUILD_NUMBER MATCHES "^([0-9]+)\.([0-9]+)\.([0-9]+)\-([0-9]+)\-.*")
        set(OpenVINO_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(OpenVINO_VERSION_MINOR ${CMAKE_MATCH_2})
        set(OpenVINO_VERSION_PATCH ${CMAKE_MATCH_3})
        set(OpenVINO_VERSION_BUILD ${CMAKE_MATCH_4})
        set(the_whole_version_is_defined_by_ci ON)
    elseif(CI_BUILD_NUMBER MATCHES "^[0-9]+$")
        set(OpenVINO_VERSION_BUILD ${CI_BUILD_NUMBER})
        # only build number is defined by CI
        set(the_whole_version_is_defined_by_ci OFF)
    elseif(CI_BUILD_NUMBER)
        message(FATAL_ERROR "Failed to parse CI_BUILD_NUMBER which is ${CI_BUILD_NUMBER}")
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

        # detect commit number
        commitNumber(OpenVINO_VERSION_BUILD_HPP)
        if(OpenVINO_VERSION_BUILD STREQUAL "000" AND DEFINED OpenVINO_VERSION_BUILD_HPP)
            set(OpenVINO_VERSION_BUILD "${OpenVINO_VERSION_BUILD_HPP}")
        else()
            set(OpenVINO_VERSION_BUILD_HPP "${OpenVINO_VERSION_BUILD}")
        endif()

        set(ov_hpp_version_is_found ON)
    endmacro()

    # detect OpenVINO version via openvino/core/version.hpp and ie_version.hpp
    ov_get_hpp_version()

    if(ov_hpp_version_is_found)
        foreach(var OpenVINO_VERSION_MAJOR OpenVINO_VERSION_MINOR OpenVINO_VERSION_PATCH OpenVINO_VERSION_BUILD)
            if(DEFINED ${var} AND NOT ${var} EQUAL ${var}_HPP)
                message(FATAL_ERROR "${var} parsed from CI_BUILD_NUMBER (${${var}}) \
                    and from openvino/core/version.hpp (${${var}_HPP}) are different")
            else()
                # CI_BUILD_NUMBER is not defined well, take info from openvino/core/version.hpp as a baseline
                set(${var} ${${var}_HPP})
            endif()
        endforeach()
    endif()

    set(OpenVINO_SOVERSION "${OpenVINO_VERSION_MAJOR}${OpenVINO_VERSION_MINOR}${OpenVINO_VERSION_PATCH}")
    string(REGEX REPLACE "^20" "" OpenVINO_SOVERSION "${OpenVINO_SOVERSION}")
    set(OpenVINO_VERSION "${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}")
    if(ENABLE_LIBRARY_VERSIONING)
        set(OpenVINO_VERSION_SUFFIX ".${OpenVINO_SOVERSION}")
    else()
        set(OpenVINO_VERSION_SUFFIX "")
    endif()
    message(STATUS "OpenVINO version is ${OpenVINO_VERSION} (Build ${OpenVINO_VERSION_BUILD})")

    if(NOT the_whole_version_is_defined_by_ci)
        # create CI_BUILD_NUMBER

        branchName(GIT_BRANCH)
        commitHash(GIT_COMMIT_HASH)

        if(NOT GIT_BRANCH STREQUAL "master")
            set(GIT_BRANCH_POSTFIX "-${GIT_BRANCH}")
        endif()

        set(CI_BUILD_NUMBER "${OpenVINO_VERSION}-${OpenVINO_VERSION_BUILD}-${GIT_COMMIT_HASH}${GIT_BRANCH_POSTFIX}")

        unset(GIT_BRANCH_POSTFIX)
        unset(GIT_BRANCH)
        unset(GIT_COMMIT_HASH)
    else()
        unset(the_whole_version_is_defined_by_ci)
    endif()
endmacro()

# provides OpenVINO version
# 1. If CI_BUILD_NUMBER is defined, parses this information
# 2. Otherwise, parses openvino/core/version.hpp
if (DEFINED ENV{CI_BUILD_NUMBER})
    set(CI_BUILD_NUMBER $ENV{CI_BUILD_NUMBER})
endif()
ov_parse_ci_build_number()

macro (addVersionDefines FILE)
    message(WARNING "'addVersionDefines' is deprecated. Please, use 'ov_add_version_defines'")

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

macro (ov_add_version_defines FILE TARGET)
    set(__version_file ${FILE})
    if(NOT IS_ABSOLUTE ${__version_file})
        set(__version_file "${CMAKE_CURRENT_SOURCE_DIR}/${__version_file}")
    endif()
    if(NOT EXISTS ${__version_file})
        message(FATAL_ERROR "${FILE} does not exists in current source directory")
    endif()
    _remove_source_from_target(${TARGET} ${FILE})
    _remove_source_from_target(${TARGET} ${__version_file})
    if (BUILD_SHARED_LIBS)
        add_library(${TARGET}_version OBJECT ${__version_file})
    else()
        add_library(${TARGET}_version STATIC ${__version_file})
    endif()
    if(SUGGEST_OVERRIDE_SUPPORTED)
        set_source_files_properties(${__version_file}
            PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
    endif()

    target_compile_definitions(${TARGET}_version PRIVATE
        CI_BUILD_NUMBER=\"${CI_BUILD_NUMBER}\"
        $<TARGET_PROPERTY:${TARGET},INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:${TARGET},COMPILE_DEFINITIONS>)
    target_include_directories(${TARGET}_version PRIVATE
        $<TARGET_PROPERTY:${TARGET},INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:${TARGET},INCLUDE_DIRECTORIES>)
    target_link_libraries(${TARGET}_version PRIVATE
        $<TARGET_PROPERTY:${TARGET},LINK_LIBRARIES>)
    target_compile_options(${TARGET}_version PRIVATE
        $<TARGET_PROPERTY:${TARGET},INTERFACE_COMPILE_OPTIONS>
        $<TARGET_PROPERTY:${TARGET},COMPILE_OPTIONS>)
    set_target_properties(${TARGET}_version
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE
        $<TARGET_PROPERTY:${TARGET},INTERPROCEDURAL_OPTIMIZATION_RELEASE>)

    target_sources(${TARGET} PRIVATE $<TARGET_OBJECTS:${TARGET}_version>)
    unset(__version_file)
endmacro()

function(ov_add_library_version library)
    if(NOT DEFINED OpenVINO_SOVERSION)
        message(FATAL_ERROR "Internal error: OpenVINO_SOVERSION is not defined")
    endif()

    if(ENABLE_LIBRARY_VERSIONING)
        set_target_properties(${library} PROPERTIES
            SOVERSION ${OpenVINO_SOVERSION}
            VERSION ${OpenVINO_VERSION})
    endif()
endfunction()
