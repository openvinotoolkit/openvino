# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME MklDnnFunctionalTests)

file(GLOB MKL_DNN_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config_param_test/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/extensions_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/network_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/normalization_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/snippet_test/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/regression_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/graph_tools/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/io_blob_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/int8_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/input_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/inference_engine_regression_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/lstm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/common_single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/ie_class/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/single_layer_tests/*.cpp
        )

list(APPEND MKL_DNN_LIBS
        IESharedTests
        inference_engine_legacy
        ${Boost_REGEX_LIBRARY})

if (USE_CNNNETWORK_LPT)
    list(APPEND ${MKL_DNN_TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/network_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instance/transformations/*.cpp)

    list(APPEND MKL_DNN_LIBS
        inference_engine_lp_transformations)
endif()

list(APPEND TEST_SRC ${MKL_DNN_TEST_SOURCES})
list(APPEND LIBRARIES ${MKL_DNN_LIBS})

list(APPEND DEPENDENCIES
        MKLDNNPlugin)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

add_executable(${TARGET_NAME}
               ${TEST_SRC}
               ${REGRESSION_TESTS}
               ${TEST_INCLUDE})

target_compile_definitions(${TARGET_NAME}
        PUBLIC ${ARGV}
        DATA_PATH=\"${DATA_PATH}\"
        MODELS_PATH=\"${MODELS_PATH}\" PARENT_SCOPE)

target_include_directories(${TARGET_NAME} PRIVATE
        ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin)

target_link_libraries(${TARGET_NAME} PRIVATE ${LIBRARIES})

add_dependencies(${TARGET_NAME} ${DEPENDENCIES})

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS CPU)
