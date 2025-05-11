# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(TARGET_NAME "tensorflow_fe_standalone_build_test")

add_library(${TARGET_NAME} STATIC standalone_build_test.cpp)

# This test verifies that application can link to TensorFlow frontend only
# Other dependencies on core header files will be resolved automatically
target_link_libraries(${TARGET_NAME} PUBLIC openvino_tensorflow_frontend)

# Enable code style check
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
