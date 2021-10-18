# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_shape_inference")

file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(SHAPE_INFER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("include" FILES ${PUBLIC_HEADERS})

# Create shared library
add_library(${TARGET_NAME} INTERFACE)

# Defines macro in C++ to load backend plugin
target_include_directories(${TARGET_NAME} INTERFACE ${SHAPE_INFER_INCLUDE_DIR} ${NGRAPH_INCLUDE_PATH})

# developer package
openvino_developer_export_targets(COMPONENT ngraph TARGETS ${TARGET_NAME})
