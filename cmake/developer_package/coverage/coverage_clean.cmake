# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED IE_COVERAGE_REPORTS)
    message(FATAL_ERROR "IE_COVERAGE_REPORTS variable is not defined")
    return()
endif()

file(REMOVE_RECURSE "${IE_COVERAGE_REPORTS}")

if(NOT DEFINED IE_COVERAGE_DIRECTORY)
    message(FATAL_ERROR "IE_COVERAGE_DIRECTORY variable is not defined")
    return()
endif()

# remove .gcno files which are kept from the previous build

file(GLOB_RECURSE gcno_files "${IE_COVERAGE_DIRECTORY}/*.gcno")
foreach(file IN LISTS gcno_files)
    string(REPLACE ".gcno" "" temp_file "${file}")
    string(REGEX REPLACE "CMakeFiles/.+dir/" "" temp_file "${temp_file}")
    string(REPLACE "${CMAKE_BINARY_DIRECTORY}" "${CMAKE_SOURCE_DIRECTORY}" source_file "${temp_file}")

    if(NOT EXISTS "${source_file}")
        file(REMOVE "${file}")
        string(REPLACE "${CMAKE_BINARY_DIRECTORY}/" "" file "${file}")
        message("Removing ${file}")
    endif()
endforeach()
