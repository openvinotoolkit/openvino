# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(REMOVE "${OUTPUT_FILE}")

set(DEFAULT_FILTER "
    -build/header_guard,\
    -build/include,\
    -build/include_order,\
    -build/include_subdir,\
    -build/include_what_you_use,\
    -build/namespaces,\
    -build/c++11,\
    -whitespace/indent,\
    -whitespace/comments,\
    -whitespace/ending_newline,\
    -runtime/references,\
    -runtime/int,\
    -runtime/explicit,\
    -readability/todo,\
    -readability/fn_size,\
")
set(FILTER "${DEFAULT_FILTER}${CUSTOM_FILTER}")

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CPPLINT_SCRIPT}"
        "--linelength=160"
        "--counting=detailed"
        "--quiet"
        "--filter=${FILTER}"
        "${INPUT_FILE}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

# Store cpplint output to file (replace problematic symbols)
string(REPLACE "\"" "&quot\;" formatted_output "${output}")
string(REPLACE "<" "&lt\;" formatted_output "${formatted_output}")
string(REPLACE ">" "&gt\;" formatted_output "${formatted_output}")
string(REPLACE "'" "&apos\;" formatted_output "${formatted_output}")
string(REPLACE "&" "&amp\;" formatted_output "${formatted_output}")
file(WRITE "${OUTPUT_FILE}" "${formatted_output}")

if(NOT SKIP_RETURN_CODE)
    # Pass through the cpplint return code
    if(NOT result EQUAL "0")
        # Display the cpplint output to console (to parse it form IDE)
        message("${output}")
        message(FATAL_ERROR "[cpplint] Code style check failed for : ${INPUT_FILE}")
    endif()
endif()
