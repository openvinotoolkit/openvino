# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
project(InferenceEngine)

set(CMAKE_MODULE_PATH "${IE_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(features_ie)

# include developer package
include(developer_package_ie)

# These options are shared with 3rdparty plugins
# by means of developer package
include(check_features_ie)

# resolving dependencies for the project
include(dependencies)

include (fuzzing)
if (ENABLE_FUZZING)
    enable_fuzzing()
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(thirdparty)

add_subdirectory(tools)

if (ENABLE_SAMPLES)
    # hint for find_package(InferenceEngine in the samples folder)
    set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")
endif()

# gflags and format_reader targets are kept inside of samples directory and
# they must be built even if samples build is disabled (required for tests and tools).
add_subdirectory(samples)

file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
add_cpplint_target(sample_cpplint
    FOR_SOURCES ${SAMPLES_SOURCES}
    EXCLUDE_PATTERNS "thirdparty/*" "pugixml/*")

if (ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

add_cpplint_report_target()
