# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME IEBehaviorSharedTests)

file(GLOB_RECURSE SHARED_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(${TARGET_NAME} STATIC ${SHARED_TESTS_SRC})
add_dependencies(${TARGET_NAME} MultiDevicePlugin inference_engine_preproc)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests")

target_link_libraries(${TARGET_NAME} PUBLIC
        funcTestUtils
        ieTestHelpers
        )

target_link_libraries(${TARGET_NAME} PUBLIC gmock)

target_include_directories(${TARGET_NAME} PUBLIC
        ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests)

# developer package
openvino_developer_export_targets(COMPONENT inference_engine_tests TARGETS ${TARGET_NAME})
