# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_preprocessor)

add_library(${TARGET_NAME} INTERFACE)

add_library(openvino::pp ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
