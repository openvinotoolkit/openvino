# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

set(TARGET_NAME InferenceEngineUnitTests)

ov_disable_deprecated_warnings()

# collect sources

file(GLOB
        TEST_SRC
        inference_engine_tests/*.cpp
        )

# TODO: fix CVS-71010 and remove BUILD_SHARED_LIBS
if (ENABLE_INTEL_GNA AND BUILD_SHARED_LIBS)
    file(GLOB
            GNA_TESTS
            engines/gna/*cpp
            engines/gna/layers/*cpp
            engines/gna/graph_tools/*cpp
            )
    source_group("gna" FILES ${GNA_TESTS})
    list(APPEND TEST_SRC ${GNA_TESTS})
    list(APPEND TEST_DEPS openvino_intel_gna_plugin_test_static)

    if(SUGGEST_OVERRIDE_SUPPORTED)
        set_source_files_properties(engines/gna/graph_tools/graph_copy_tests.cpp
                                    engines/gna/graph_tools/graph_tools_test.cpp
                                    PROPERTIES COMPILE_OPTIONS -Wno-suggest-override)
    endif()

    # TODO: fix GNA tests
    if(OFF)
        set(gna_stub "${CMAKE_CURRENT_SOURCE_DIR}/engines/gna/gna_api_stub.cpp")
        list(REMOVE_ITEM TEST_SRC "${gna_stub}")
        add_library(libGNAStubs SHARED "${gna_stub}")

        find_package(libGNA NO_MODULE REQUIRED)

        target_include_directories(libGNAStubs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/engines/gna/")
        target_link_libraries(libGNAStubs PUBLIC libGNA::API gmock gtest)
        if(UNIX)
            set_target_properties(libGNAStubs PROPERTIES CXX_VISIBILITY_PRESET default)
        endif()
        list(APPEND TEST_DEPS libGNAStubs)
    endif()
endif()

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

# create target

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE})

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# because IE unit tests use plugin and IE object files compiled with LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

## Mock macros doesn't use "override" specificator
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>: -Wno-error=inconsistent-missing-override >)
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:AppleClang>: -Wno-error=inconsistent-missing-override >)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS -IGNORE:4286)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    # static libraries
    inference_engine_s # need to have this explicitly for USE_STATIC_IE
    unit_test_utils
    ieTestHelpers_s
    ${TEST_DEPS}

    # dynamic libraries
    inference_engine_transformations
    inference_engine_lp_transformations
    )

ov_try_use_gold_linker()

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS OV UNIT)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

if(TARGET libGNAStubs)
    install(TARGETS libGNAStubs
            RUNTIME DESTINATION tests
            COMPONENT tests
            EXCLUDE_FROM_ALL)
endif()

add_dependencies(${TARGET_NAME} mock_engine)
