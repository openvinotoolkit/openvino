# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(REMOVE "${OUTPUT_FILE}")

execute_process(
    COMMAND
        "${PYTHON_EXECUTABLE}"
        "${CPPLINT_SCRIPT}"
        "--linelength=160"
        "--counting=detailed"
        "--filter=-readability/fn_size"
        "${INPUT_FILE}"
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

# Display the cpplint output to console (to parse it form IDE)
message("${output}")

# Store cpplint output to file (replace problematic symbols)
string(REPLACE "\"" "&quot\;" output "${output}")
string(REPLACE "<" "&lt\;" output "${output}")
string(REPLACE ">" "&gt\;" output "${output}")
string(REPLACE "'" "&apos\;" output "${output}")
string(REPLACE "&" "&amp\;" output "${output}")
file(WRITE "${OUTPUT_FILE}" "${output}")

if(NOT SKIP_RETURN_CODE)
    # Pass through the cpplint return code
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "[cpplint] Code style check failed for : ${INPUT_FILE}")
    endif()
endif()
