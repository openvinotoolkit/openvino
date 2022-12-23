# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_transformations_tests)

ov_add_test_target(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDENCIES
    LINK_LIBRARIES
        gmock
        funcTestUtils
        offline_transformations
        sharedTestClasses
        lptNgraphFunctions
    ADD_CLANG_FORMAT
    INCLUDES
        $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
    LABELS
        TRANSFORMATIONS
)

file(GLOB_RECURSE legacy_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/legacy/*.cpp" # CVS-55385
    )

set_source_files_properties(${legacy_tests} PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

if(WIN32)
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()
