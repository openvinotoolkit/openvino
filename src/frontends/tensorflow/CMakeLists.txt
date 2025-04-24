# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(src)

if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} 
                       ${CMAKE_CURRENT_SOURCE_DIR}/docs/check_supported_ops.py
                       ${CMAKE_CURRENT_SOURCE_DIR}/src/op_table.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/docs/supported_ops.md
        RESULT_VARIABLE SCRIPT_RESULT
        OUTPUT_VARIABLE SCRIPT_OUTPUT
        ERROR_VARIABLE SCRIPT_ERROR
    )

    if(NOT SCRIPT_RESULT EQUAL 0)
        message(FATAL_ERROR "Python script failed with return code ${SCRIPT_RESULT}\nOutput: ${SCRIPT_OUTPUT}\nError: ${SCRIPT_ERROR}")
    endif()
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
