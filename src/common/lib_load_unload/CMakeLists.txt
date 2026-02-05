# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_lib_load_unload)

# Main shutdown library without weak symbols
add_library(${TARGET_NAME} OBJECT src/lib_load_unload.cpp)
add_library(openvino::lib_load_unload ALIAS ${TARGET_NAME})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME openvino_lib_load_unload)


# Separate library for weak default implementations
set(WEAK_TARGET_NAME openvino_lib_load_unload_weak)
add_library(${WEAK_TARGET_NAME} STATIC src/load_unload_weak_defs.cpp)
target_include_directories(${WEAK_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(${WEAK_TARGET_NAME} PROPERTIES EXPORT_NAME openvino_lib_load_unload_weak)

target_link_libraries(${TARGET_NAME} PRIVATE ${WEAK_TARGET_NAME})
