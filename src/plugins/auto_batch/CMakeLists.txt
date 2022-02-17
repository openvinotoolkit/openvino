# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_auto_batch_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "BATCH"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR auto_batch.cpp ADD_CLANG_FORMAT)

target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
