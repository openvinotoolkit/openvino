# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT WIN32)
    function(ie_add_manifest)
    endfunction()
    return()
endif()

set(UWP_SDK_PATH "${PROGRAMFILES}/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64")
find_host_program(MANIFEST_TOOL NAMES mt PATHS ${UWP_SDK_PATH} NO_DEFAULT_PATH)
if(NOT MANIFEST_TOOL)
    message(FATAL_ERROR "Manifest cannot be embedded since mt.exe is not found")
endif()
unset(UWP_SDK_PATH)

#
# ie_add_manifest(TARGET_NAME <target name>
#                 [DEPENDENCIES <dependencies>])
#
macro(ie_add_manifest)
    include(CMakeParseArguments)
    cmake_parse_arguments(MANIFEST "" "TARGET_NAME" "DEPENDENCIES" ${ARGN})

    if(NOT TARGET ${MANIFEST_TARGET_NAME})
        message(FATAL_ERROR "${MANIFEST_TARGET_NAME} does not represent a target")
    endif()
    
    get_target_property(MANIFEST_TARGET_TYPE ${MANIFEST_TARGET_NAME} TYPE)
    set(MANIFEST_GENERATE_SCRIPT "${OpenVINO_MAIN_SOURCE_DIR}/cmake/manifests/manifests_generate.cmake")
    set(MANIFEST_TOKEN "0000000000000000")
    set(MANIFEST_VERSION "2021.3.0.0")
    set(MANIFEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/${MANIFEST_TARGET_NAME}.dll.manifest")
    string(REPLACE ";" "." MANIFEST_DEPENDENCIES_STR "${MANIFEST_DEPENDENCIES}")

    add_custom_command(TARGET ${MANIFEST_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -D MANIFEST_TOOL=${MANIFEST_TOOL}
            -D MANIFEST_FILE=${MANIFEST_FILE}
            -D MANIFEST_TARGET_FILE=$<TARGET_FILE:${MANIFEST_TARGET_NAME}>
            -D MANIFEST_TARGET_NAME=${MANIFEST_TARGET_NAME}
            -D MANIFEST_DEPENDENCIES=${MANIFEST_DEPENDENCIES_STR}
            -D MANIFEST_TOKEN=${MANIFEST_TOKEN}
            -D MANIFEST_VERSION=${MANIFEST_VERSION}
            -D MANIFEST_TARGET_TYPE=${MANIFEST_TARGET_TYPE}
            -P ${MANIFEST_GENERATE_SCRIPT}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Embed manifest to ${MANIFEST_TARGET_NAME}"
        VERBATIM)

    foreach(var MANIFEST_TOKEN MANIFEST_VERSION MANIFEST_GENERATE_SCRIPT
                MANIFEST_TARGET_NAME MANIFEST_DEPENDENCIES MANIFEST_FILE
                MANIFEST_DEPENDENCIES_STR MANIFEST_TARGET_TYPE)
        unset("${var}")
    endforeach()
endmacro()
