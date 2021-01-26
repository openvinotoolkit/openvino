# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

set(TARGET_NAME InferenceEngineUnitTests)

disable_deprecated_warnings()

# rpath enabled for unit tests only
SET (CMAKE_SKIP_RPATH OFF)

# collect sources

file(GLOB
        TEST_SRC
        graph_tools/*.cpp
        inference_engine_tests/*.cpp
        )

if (ENABLE_GNA)
    file(GLOB
            GNA_TESTS
            engines/gna/*cpp
            engines/gna/layers/*cpp
            )
    list(APPEND TEST_SRC ${GNA_TESTS})
    source_group("gna" FILES ${GNA_TESTS})

    # TODO: fix GNA tests
    if(OFF)
        set(gna_stub "${CMAKE_CURRENT_SOURCE_DIR}/engines/gna/gna_api_stub.cpp")
        list(REMOVE_ITEM TEST_SRC "${gna_stub}")
        add_library(libGNAStubs SHARED "${gna_stub}")

        find_package(libGNA)
        if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
            set(GNA_LIBRARY_VERSION_NUMBER 2)
        else()
            set(GNA_LIBRARY_VERSION_NUMBER 1)
        endif()

        target_include_directories(libGNAStubs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/engines/gna/")
        target_compile_definitions(libGNAStubs PUBLIC GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER})
        target_link_libraries(libGNAStubs PUBLIC libGNA::API gmock gtest)
        if(UNIX)
            set_target_properties(libGNAStubs PROPERTIES CXX_VISIBILITY_PRESET default)
        endif()
    endif()

    set (GNA_TEST_ENGINE GNAPlugin_test_static)
endif()

if (ENABLE_MKL_DNN)
    if (GEMM STREQUAL "MKL")
        add_definitions(-DUSE_MKL)
    endif ()
    file(GLOB
            MKLDNN_TESTS
            engines/mkldnn/*.cpp
            engines/mkldnn/graph/layers/extensions/*.cpp
            engines/mkldnn/graph/layers/internal/*.cpp
            engines/mkldnn/graph/structure/*.cpp
            engines/mkldnn/graph/*.cpp)
    file(GLOB
            MKLDNN_TESTS_INCLUDE engines/mkldnn/graph/*.hpp)
    include_directories(
            ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include
            engines/mkldnn/graph
            ${CMAKE_BINARY_DIR}/include/)

    source_group("mkldnn" FILES ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE})
    list(APPEND MKLDNN_TESTS $<TARGET_OBJECTS:MKLDNNPlugin_obj>)
endif ()

if (ENABLE_MYRIAD)
    include(${XLINK_DIR}/XLink.cmake)

    file(GLOB
            VPU_TESTS
            engines/vpu/*cpp
            engines/vpu/mvnc/*cpp
            engines/vpu/myriad_tests/*cpp
            engines/vpu/myriad_tests/helpers/*cpp
            engines/vpu/myriad_tests/helpers/*h
            ${IE_MAIN_SOURCE_DIR}/src/vpu/myriad_plugin/*.cpp
            )
    include_directories(
            engines/vpu/myriad_tests/helpers
            ${IE_MAIN_SOURCE_DIR}/src/vpu/myriad_plugin
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc/include/watchdog
            ${XLINK_INCLUDE}
            ${XLINK_PLATFORM_INCLUDE})


    list(APPEND TEST_SRC ${VPU_TESTS})
    source_group("vpu" FILES ${VPU_TESTS})
endif()

file(GLOB
        TEST_INCLUDE
        shape_infer/*.hpp)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

# create target

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE} ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE} ${DLIA_TESTS})
set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        ${IE_MAIN_SOURCE_DIR}/src/gna_plugin
        "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# because IE unit tests use plugin and IE object files compiled with LTO
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
    set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

## Mock macros doesn't use "override" specificator
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>: -Wno-error=inconsistent-missing-override >)
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:AppleClang>: -Wno-error=inconsistent-missing-override >)

if (ENABLE_MYRIAD)
    target_compile_definitions(${TARGET_NAME} PRIVATE __PC__)
    target_link_libraries(${TARGET_NAME} PRIVATE mvnc vpu_graph_transformer_test_static)
endif ()

target_link_libraries(${TARGET_NAME} PRIVATE
    # static libraries
    inference_engine_s # need to have this explicitly for USE_STATIC_IE
    unitTestUtils
    ieTestHelpers_s
    ${GNA_TEST_ENGINE}

    # dynamic libraries
    inference_engine_transformations
    inference_engine_lp_transformations)

if(TARGET libGNAStubs)
    target_link_libraries(${TARGET_NAME} PRIVATE libGNAStubs)
endif()

if (ENABLE_MKL_DNN)
    target_link_libraries(${TARGET_NAME} PRIVATE mkldnn)
endif ()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS IE)

add_dependencies(${TARGET_NAME} mock_engine)
