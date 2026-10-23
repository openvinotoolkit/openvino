# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Standalone CMake script (invoked via `cmake -P`) that installs a single CPack
# component into an isolated staging directory and records the resulting relative
# file list as a BOM (bill of materials) manifest.
#
# Expected variables (passed via -D on the command line):
#   OV_BOM_COMPONENT     - name of the CPack component to install
#   OV_BOM_BUILD_DIR     - path to the CMake build directory to install from
#   OV_BOM_STAGING_DIR   - scratch directory used as the install --prefix
#   OV_BOM_OUTPUT_FILE   - path of the resulting BOM JSON file
#   OV_BOM_CONFIG        - (optional) build configuration, for multi-config generators
#

foreach(_var OV_BOM_COMPONENT OV_BOM_BUILD_DIR OV_BOM_STAGING_DIR OV_BOM_OUTPUT_FILE)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "bom_generate.cmake: '${_var}' must be defined")
    endif()
endforeach()

# start from a clean staging directory, so files removed since the previous run do not linger
file(REMOVE_RECURSE "${OV_BOM_STAGING_DIR}")
file(MAKE_DIRECTORY "${OV_BOM_STAGING_DIR}")

set(_ov_bom_install_command
    "${CMAKE_COMMAND}" --install "${OV_BOM_BUILD_DIR}"
    --prefix "${OV_BOM_STAGING_DIR}"
    --component "${OV_BOM_COMPONENT}")

if(OV_BOM_CONFIG)
    list(APPEND _ov_bom_install_command --config "${OV_BOM_CONFIG}")
endif()

execute_process(
    COMMAND ${_ov_bom_install_command}
    RESULT_VARIABLE _ov_bom_install_result
    OUTPUT_VARIABLE _ov_bom_install_output
    ERROR_VARIABLE _ov_bom_install_output)

if(NOT _ov_bom_install_result EQUAL 0)
    message(FATAL_ERROR
        "bom_generate.cmake: failed to install CPack component '${OV_BOM_COMPONENT}':\n${_ov_bom_install_output}")
endif()

# collect the resulting file layout, normalized to portable, sorted, forward-slash relative paths
file(GLOB_RECURSE _ov_bom_files
     LIST_DIRECTORIES FALSE
     RELATIVE "${OV_BOM_STAGING_DIR}"
     "${OV_BOM_STAGING_DIR}/*")

set(_ov_bom_normalized_files)
foreach(_file IN LISTS _ov_bom_files)
    file(TO_CMAKE_PATH "${_file}" _file)
    list(APPEND _ov_bom_normalized_files "${_file}")
endforeach()
list(REMOVE_DUPLICATES _ov_bom_normalized_files)
list(SORT _ov_bom_normalized_files CASE SENSITIVE)

get_filename_component(_ov_bom_output_dir "${OV_BOM_OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_ov_bom_output_dir}")

list(LENGTH _ov_bom_normalized_files _ov_bom_files_count)

# build the JSON manifest manually (plain list of relative file paths) to avoid extra dependencies
set(_ov_bom_json "{\n  \"component\": \"${OV_BOM_COMPONENT}\",\n  \"files\": [")
if(_ov_bom_files_count GREATER 0)
    string(APPEND _ov_bom_json "\n")
    set(_ov_bom_index 0)
    foreach(_file IN LISTS _ov_bom_normalized_files)
        math(EXPR _ov_bom_index "${_ov_bom_index} + 1")
        string(REPLACE "\\" "\\\\" _file "${_file}")
        string(REPLACE "\"" "\\\"" _file "${_file}")
        if(_ov_bom_index EQUAL _ov_bom_files_count)
            string(APPEND _ov_bom_json "    \"${_file}\"\n")
        else()
            string(APPEND _ov_bom_json "    \"${_file}\",\n")
        endif()
    endforeach()
endif()
string(APPEND _ov_bom_json "  ]\n}\n")

file(WRITE "${OV_BOM_OUTPUT_FILE}" "${_ov_bom_json}")
file(REMOVE_RECURSE "${OV_BOM_STAGING_DIR}")

message(STATUS "[BOM] '${OV_BOM_COMPONENT}': ${_ov_bom_files_count} file(s) -> ${OV_BOM_OUTPUT_FILE}")
