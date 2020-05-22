# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

disable_deprecated_warnings()

set(TARGET_NAME TemplateBehaviorTests)

file(GLOB_RECURSE TEST_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

file(GLOB_RECURSE TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

list(APPEND DEPENDENCIES
        templatePlugin)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

add_executable(${TARGET_NAME}
        ${TEST_SRC}
        ${TEST_INCLUDE})

target_link_libraries(${TARGET_NAME} PRIVATE IE::IEBehaviorSharedTests)

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} ${DEPENDENCIES})

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
