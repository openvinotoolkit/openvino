# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_shutdown)

# Use OBJECT instead of STATIC to ensure DLLMain is invoked when the shared library is unloaded
add_library(${TARGET_NAME} OBJECT shutdown.cpp)

add_library(openvino::shutdown ALIAS ${TARGET_NAME})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::itt)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME openvino_shutdown)
