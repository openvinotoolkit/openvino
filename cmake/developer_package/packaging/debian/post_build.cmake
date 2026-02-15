# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_program(lintian_PROGRAM NAMES lintian DOC "Path to lintian tool")
if(NOT lintian_PROGRAM)
    message(WARNING "Failed to find 'lintian' tool, use 'sudo apt-get install lintian' to install it")
    return()
endif()

execute_process(COMMAND "${lintian_PROGRAM}" --version
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                RESULT_VARIABLE lintian_code
                OUTPUT_VARIABLE lintian_version)

if(NOT lintian_code EQUAL 0)
    message(FATAL_ERROR "Internal error: Failed to determine lintian version")
else()
    message(STATUS "${lintian_version}")
endif()

set(lintian_passed ON)

foreach(deb_file IN LISTS CPACK_PACKAGE_FILES)
    execute_process(COMMAND "${lintian_PROGRAM}" ${deb_file}
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                    RESULT_VARIABLE lintian_exit_code
                    OUTPUT_VARIABLE lintian_output)

    get_filename_component(deb_name "${deb_file}" NAME)

    if(NOT lintian_exit_code EQUAL 0)
        message("Package ${deb_name}:")
        message("${lintian_output}")
        set(lintian_passed OFF)
    endif()
endforeach()

if(NOT lintian_passed)
    message(FATAL_ERROR "Lintian has found some mistakes")
endif()
