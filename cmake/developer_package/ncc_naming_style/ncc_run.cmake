# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var NCC_PY_SCRIPT Python3_EXECUTABLE OUTPUT_FILE DEFINITIONS EXPECTED_FAIL
    INPUT_FILE ADDITIONAL_INCLUDE_DIRECTORIES STYLE_FILE CLANG_LIB_PATH)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined for ncc_run.cmake")
    endif()
endforeach()

file(REMOVE "${OUTPUT_FILE}")

if(DEFINITIONS)
    set(defs --definition ${DEFINITIONS})
endif()

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        "${NCC_PY_SCRIPT}"
        --path ${INPUT_FILE}
        --style ${STYLE_FILE}
        --clang-lib ${CLANG_LIB_PATH}
        ${defs}
        --include ${ADDITIONAL_INCLUDE_DIRECTORIES}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

file(WRITE "${OUTPUT_FILE}" "${output}")

if(NOT result EQUAL "0")
    set(failed ON)
endif()

if(EXPECTED_FAIL AND NOT failed)
    message(FATAL_ERROR "[ncc self check] Self check is not failed for ${INPUT_FILE}")
endif()

if(failed AND NOT EXPECTED_FAIL)
    # Display the output to console (to parse it form IDE)
    message("${output}\n${error}")
    message(FATAL_ERROR  "[ncc naming style] Naming style check failed for ${INPUT_FILE}")
endif()
