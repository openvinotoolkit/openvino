# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var NCC_PY_SCRIPT PYTHON_EXECUTABLE OUTPUT_FILE
    INPUT_FILE ADDITIONAL_INCLUDE_DIRECTORIES STYLE_FILE)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined for ncc_run.cmake")
    endif()
endforeach()

file(REMOVE "${OUTPUT_FILE}")

execute_process(
    COMMAND
        "${PYTHON_EXECUTABLE}"
        "${NCC_PY_SCRIPT}"
        --path ${INPUT_FILE}
        --style ${STYLE_FILE}
        --include ${ADDITIONAL_INCLUDE_DIRECTORIES}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

file(WRITE "${OUTPUT_FILE}" "${output}")

if(NOT result EQUAL "0")
    # Display the output to console (to parse it form IDE)
    message("${output}")
    message(FATAL_ERROR  "[ncc naming style] Naming style check failed for ${INPUT_FILE}")
endif()
