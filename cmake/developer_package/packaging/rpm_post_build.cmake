# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_program(rpmlint_PROGRAM NAMES rpmlint DOC "Path to rpmlint tool")
if(NOT rpmlint_PROGRAM)
    message(WARNING "Failed to find 'rpmlint' tool, use 'sudo dnf install rpmlint' to install it")
    return()
endif()

execute_process(COMMAND "${rpmlint_PROGRAM}" --version
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                RESULT_VARIABLE rpmlint_code
                OUTPUT_VARIABLE rpmlint_version)

if(NOT rpmlint_code EQUAL 0)
    message(FATAL_ERROR "Internal error: Failed to determine rpmlint version")
else()
    message(STATUS "${rpmlint_version}")
endif()

set(rpmlint_passed ON)

foreach(rpm_file IN LISTS CPACK_PACKAGE_FILES)
    execute_process(COMMAND "${rpmlint_PROGRAM}" ${rpm_file}
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                    RESULT_VARIABLE rpmlint_exit_code
                    OUTPUT_VARIABLE rpmlint_output)

    get_filename_component(rpm_name "${rpm_file}" NAME)

    if(NOT rpmlint_exit_code EQUAL 0)
        message("Package ${rpm_name}:")
        message("${rpmlint_output}")
        set(rpmlint_passed OFF)
    endif()
endforeach()

if(NOT rpmlint_passed)
    message(FATAL_ERROR "rpmlint has found some mistakes")
endif()
