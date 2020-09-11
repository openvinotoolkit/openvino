# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ie_tests)

file(GLOB TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# create target

add_library(${TARGET_NAME} STATIC ${TEST_INCLUDE} ${TEST_SRC})

list(APPEND EXPORT_DEPENDENCIES
        funcTestUtils
        ieTestHelpers
        )

target_include_directories(${TARGET_NAME}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME} PUBLIC
        format_reader
        ${EXPORT_DEPENDENCIES}
        )

# developer package

ie_developer_export_targets(${TARGET_NAME} ${EXPORT_DEPENDENCIES})
