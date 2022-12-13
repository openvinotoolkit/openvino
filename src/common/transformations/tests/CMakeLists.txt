# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_transformations_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4305)
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4050)
    ie_add_compiler_flags(/wd4250)
    ie_add_compiler_flags(/wd4334)
    ie_add_compiler_flags(/wd4661)
    ie_add_compiler_flags(/wd4273)
    ie_add_compiler_flags(/wd4309)
    ie_add_compiler_flags(/wd4804)
endif()
if(NOT MSVC)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-sign-compare)
    ie_add_compiler_flags(-Wno-odr)
    ie_add_compiler_flags(-Wno-all)
endif()

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
    INCLUDES
        $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
    ADD_CPPLINT
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
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()
