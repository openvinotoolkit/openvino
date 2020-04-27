# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME IEBehaviorSharedTests)

disable_deprecated_warnings()

file(GLOB_RECURSE SHARED_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_library(${TARGET_NAME} STATIC ${SHARED_TESTS_SRC})
add_dependencies(${TARGET_NAME} inference_engine_preproc)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests")

target_link_libraries(${TARGET_NAME} PUBLIC
        funcTestUtils
        ieTestHelpers
        )

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC gmock)

target_include_directories(${TARGET_NAME} PUBLIC
        ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp_wrappers)

# developer package
ie_developer_export_targets(${TARGET_NAME})
