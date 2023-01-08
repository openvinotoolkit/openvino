# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME IEBehaviorSharedTests)

file(GLOB_RECURSE SHARED_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(${TARGET_NAME} STATIC ${SHARED_TESTS_SRC})
add_dependencies(${TARGET_NAME} openvino_gapi_preproc)

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    add_dependencies(${TARGET_NAME} openvino_auto_batch_plugin)
endif()


target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests")

target_link_libraries(${TARGET_NAME} PUBLIC
        funcTestUtils
        inference_engine_legacy
        ieTestHelpers
        )

target_link_libraries(${TARGET_NAME} PUBLIC gmock)

target_include_directories(${TARGET_NAME} PUBLIC
        $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests)

# developer package
openvino_developer_export_targets(COMPONENT tests TARGETS ${TARGET_NAME})
