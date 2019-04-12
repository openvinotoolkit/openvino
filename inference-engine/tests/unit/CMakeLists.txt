# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

set(TARGET_NAME InferenceEngineUnitTests)

#rpath enabled for unit tests only
SET (CMAKE_SKIP_RPATH OFF)

# collect sources

file(GLOB
        TEST_SRC
        graph_tools/*.cpp
        inference_engine_tests/*.cpp
        inference_engine_tests/cpp_interfaces/*.cpp
        mem_solver/*.cpp
        cnn_network/*.cpp
        builders/*.cpp
        transformations/*.cpp
        shape_infer/*.cpp
        shape_infer/built-in/*.cpp
        topology_verification_tests/*.cpp
        stress_tests/*.cpp
        )

if (ENABLE_GNA)
    file(GLOB
            GNA_TESTS
            engines/gna/*cpp
            )
    list(APPEND TEST_SRC ${GNA_TESTS})
    source_group("gna" FILES ${GNA_TESTS})

    find_package(libGNA)
    include_directories(${libGNA_INCLUDE_DIRS})

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
endif ()

file(GLOB
        TEST_INCLUDE
        shape_infer/*.hpp)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

# create target

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE} ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE} ${DLAI_TESTS} transformations/sub_test.cpp transformations/tranformations_test.hpp)
set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        ${IE_MAIN_SOURCE_DIR}/src/gna_plugin
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/src/extension
        ${IE_MAIN_SOURCE_DIR}/src/extension/common
        "${CMAKE_CURRENT_SOURCE_DIR}/mocks")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

## Mock macros doesn't use "override" specificator
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>: -Wno-inconsistent-missing-override >)
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:AppleClang>: -Wno-inconsistent-missing-override >)

target_link_libraries(${TARGET_NAME} PRIVATE
    gtest
    gtest_main
    gmock
    gflags
    inference_engine_s
    helpers
    ${CMAKE_DL_LIBS}
    ${GNA_TEST_ENGINE})

add_dependencies(${TARGET_NAME} ie_cpu_extension)

if (ENABLE_MKL_DNN)
    target_link_libraries(${TARGET_NAME} PRIVATE
            test_MKLDNNPlugin
            mkldnn)
endif ()

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} mock_engine)

# GAPI unit tests
add_subdirectory(opencv_test_gapi)
