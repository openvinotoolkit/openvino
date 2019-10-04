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
        http_client/*.cpp
        inference_engine_tests/*.cpp
        inference_engine_tests/cpp_interfaces/*.cpp
        inference_engine_tests/normalization/*.cpp
        cnn_network/*.cpp
        builders/*.cpp
        # TODO: apeskov: Please fix issue CVS
        # shape_infer/*.cpp
        shape_infer/built-in/*.cpp
        topology_verification_tests/*.cpp
        stress_tests/*.cpp
        cpp_api/*.cpp
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
            engines/mkldnn/normalizer/*.cpp
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

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE} ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE} ${DLIA_TESTS})
set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        ${IE_MAIN_SOURCE_DIR}/src/gna_plugin
        ${IE_MAIN_SOURCE_DIR}/src/extension
        ${IE_MAIN_SOURCE_DIR}/src/extension/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/ngraph/src
        "${CMAKE_CURRENT_SOURCE_DIR}/mocks")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

## Mock macros doesn't use "override" specificator
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:Clang>: -Wno-inconsistent-missing-override >)
target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:AppleClang>: -Wno-inconsistent-missing-override >)

if (ENABLE_MYRIAD)
    target_link_libraries(${TARGET_NAME} PRIVATE mvnc vpu_graph_transformer_test_static)
endif ()

target_link_libraries(${TARGET_NAME} PRIVATE
    gtest
    gflags
    gtest_main
    gmock
    ngraph
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

if (ENABLE_DLIA)
    target_link_libraries(${TARGET_NAME} PRIVATE dliaPluginIOTransformations)
endif ()

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} mock_engine)
