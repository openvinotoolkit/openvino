# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME MklDnnBehaviorTests)

file(GLOB_RECURSE TEST_INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

file(GLOB_RECURSE TEST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

list(APPEND DEPENDENCIES
        MKLDNNPlugin)

source_group("src" FILES ${TEST_SRC})
source_group("include" FILES ${TEST_INCLUDE})

add_executable(${TARGET_NAME}
        ${TEST_SRC}
        ${TEST_INCLUDE})

target_link_libraries(${TARGET_NAME} PRIVATE IEBehaviorSharedTests)

target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:inference_engine_preproc,INTERFACE_INCLUDE_DIRECTORIES> ${IE_MAIN_SOURCE_DIR}/samples)

add_test(NAME ${TARGET_NAME}
        COMMAND ${TARGET_NAME})

add_dependencies(${TARGET_NAME} ${DEPENDENCIES})