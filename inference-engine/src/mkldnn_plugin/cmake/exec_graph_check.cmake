# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED REF_GRAPH)
    message(FATAL_ERROR "Reference execution graph is not defined")
endif()

if(NOT DEFINED ACTUAL_GRAPH)
    message(FATAL_ERROR "Actual execution graph is not defined")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${REF_GRAPH}" "${ACTUAL_GRAPH}"
                RESULT_VARIABLE compare_result)

if( compare_result EQUAL 0)
    message("The files are identical.")
elseif( compare_result EQUAL 1)
    message("The files are different.")
else()
    message("Error while comparing the files.")
endif()
