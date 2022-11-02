# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var PYTHON_EXECUTABLE WORKING_DIRECTORY REPORT_FILE WHEEL_VERSION PACKAGE_FILE)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

# find programs

find_program(fdupes_PROGRAM NAMES fdupes DOC "Path to fdupes")
if(NOT fdupes_PROGRAM)
    message(WARNING "Failed to find 'fdupes' tool, use 'sudo apt-get install fdupes' to install it")
    return()
endif()

# execute

get_filename_component(wheel_name "${PACKAGE_FILE}" NAME)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -m wheel unpack ${PACKAGE_FILE}
                WORKING_DIRECTORY ${WORKING_DIRECTORY}
                OUTPUT_VARIABLE output_message
                ERROR_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT exit_code EQUAL 0)
    message(FATAL_ERROR "Failed to unpack wheel package")
endif()

set(WORKING_DIRECTORY "${WORKING_DIRECTORY}/openvino-${WHEEL_VERSION}")
if(NOT EXISTS "${WORKING_DIRECTORY}")
    message(FATAL_ERROR "Failed to find ${WORKING_DIRECTORY}")
endif()

execute_process(COMMAND ${fdupes_PROGRAM} -f -r "${WORKING_DIRECTORY}"
                OUTPUT_VARIABLE output_message
                ERROR_VARIABLE error_message
                RESULT_VARIABLE exit_code
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# remove unpacked directory
file(REMOVE_RECURSE "${WORKING_DIRECTORY}")

# write output

file(WRITE "${REPORT_FILE}" "${output_message}")

if(output_message)
    message(FATAL_ERROR "${output_message}")
endif()
