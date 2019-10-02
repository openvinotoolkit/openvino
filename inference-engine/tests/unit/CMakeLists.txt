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

if (ENABLE_NGRAPH)
    file(GLOB NGRAPH_TESTS
        inference_engine_tests/ngraph_reader_tests/*.cpp
        inference_engine_tests/ngraph_reader_tests/*.h)
    list(APPEND TEST_SRC ${NGRAPH_TESTS})
    source_group("ngraph" FILES ${NGRAPH_TESTS})
else()
    list(REMOVE_ITEM TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/inference_engine_tests/ngraph_reshape_tests.cpp)
endif()

# disable deprecated warnings for NN Builder

function(nn_builder_disable_warnings)
    disable_deprecated_warnings()
    file(GLOB NN_BUILDER_TESTS_SRC builders/*.cpp)
    set_source_files_properties(${NN_BUILDER_TESTS_SRC} PROPERTIES COMPILE_FLAGS ${ie_cxx_deprecated})
endfunction()

nn_builder_disable_warnings()

if (ENABLE_GNA)
    file(GLOB
            GNA_TESTS
            engines/gna/*cpp
            )
    list(APPEND TEST_SRC ${GNA_TESTS})
    source_group("gna" FILES ${GNA_TESTS})

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
            ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
            ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
            engines/mkldnn/graph
            ${CMAKE_BINARY_DIR}/include/)

    source_group("mkldnn" FILES ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE})

    foreach(obj_lib IN LISTS mkldnn_plugin_object_libraries)
        message("${obj_lib}")
        list(APPEND mkldnn_object_files $<TARGET_OBJECTS:${obj_lib}>)
    endforeach()
    list(APPEND MKLDNN_TESTS ${mkldnn_object_files})
endif ()

if (ENABLE_MYRIAD)
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
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/XLink/shared
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/watchdog
            ${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/WinPthread
    )

    list(APPEND TEST_SRC ${VPU_TESTS})
    source_group("vpu" FILES ${VPU_TESTS})
endif()

file(GLOB
        TEST_INCLUDE
        shape_infer/*.hpp)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

# create target

add_executable(${TARGET_NAME} ${TEST_SRC} ${TEST_INCLUDE} ${MKLDNN_TESTS} ${MKLDNN_TESTS_INCLUDE})
set_ie_threading_interface_for(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        ${IE_MAIN_SOURCE_DIR}/src/gna_plugin
        ${NGRAPH_INCLUDE_DIR}
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
    ${NGRAPH_LIBRARY}
    inference_engine_preproc_s
    helpers
    ${CMAKE_DL_LIBS}
    ${GNA_TEST_ENGINE})

if (ENABLE_MKL_DNN)
    target_link_libraries(${TARGET_NAME} PRIVATE mkldnn)
endif ()

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} mock_engine)
