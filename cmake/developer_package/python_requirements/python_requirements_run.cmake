
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --upgrade pip
    RESULT_VARIABLE result_code
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var)

if(result_code EQUAL 0)
    file(WRITE "${OUTPUT_FILE}" "${output_var}")
else()
    message(WARNING "Failed to upgrade pip: ${error_var} ${output_var}")
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install --upgrade pip
    COMMAND "${PYTHON_EXECUTABLE}" -m pip install -r "${REQUIREMENTS_FILE}"
    RESULT_VARIABLE result_code
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var)

if(result_code EQUAL 0)
    file(APPEND "${OUTPUT_FILE}" "${output_var}")
else()
    message(WARNING "Failed to install ${REQUIREMENTS_FILE}: ${error_var} ${output_var}")
endif()
