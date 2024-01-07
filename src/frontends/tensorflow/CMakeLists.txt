# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(src)

execute_process(
    COMMAND ${Python3_EXECUTABLE} 
                   ${CMAKE_CURRENT_SOURCE_DIR}/docs/check_supported_ops.py
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/op_table.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/docs/supported_ops.md
    RESULT_VARIABLE SCRIPT_RESULT
    OUTPUT_VARIABLE SCRIPT_OUTPUT
    ERROR_VARIABLE SCRIPT_ERROR
)

# Check the result and handle exceptions
if(SCRIPT_RESULT EQUAL 0)
    message(STATUS "Python script executed successfully")
else()
    message(FATAL_ERROR "Python script failed with return code ${SCRIPT_RESULT}\nOutput: ${SCRIPT_OUTPUT}\nError: ${SCRIPT_ERROR}")
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
