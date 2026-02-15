# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED OV_COVERAGE_OUTPUT_FILE)
    message(FATAL_ERROR "OV_COVERAGE_OUTPUT_FILE is not defined")
endif()

if(NOT DEFINED OV_COVERAGE_INPUT_FILES)
    message(FATAL_ERROR "OV_COVERAGE_INPUT_FILES is not defined")
endif()

set(command lcov --quiet)
foreach(input_info_file IN LISTS OV_COVERAGE_INPUT_FILES)
    file(SIZE ${input_info_file} size)
    if(NOT size EQUAL 0)
        list(APPEND command --add-tracefile "${input_info_file}")
    endif()
endforeach()
list(APPEND command --output-file ${OV_COVERAGE_OUTPUT_FILE})

execute_process(COMMAND ${command})
