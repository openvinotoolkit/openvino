#
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME level_zero_wrapper)

add_library(${TARGET_NAME} SHARED
    level_zero_wrapper.h
    level_zero_wrapper.cpp
)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(${TARGET_NAME}
  PUBLIC
    $<TARGET_PROPERTY:LevelZero::Headers,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:LevelZero::NPUExt,INTERFACE_INCLUDE_DIRECTORIES>) 

target_link_libraries(${TARGET_NAME} PRIVATE ze_loader)
