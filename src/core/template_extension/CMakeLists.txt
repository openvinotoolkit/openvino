# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(old)
add_subdirectory(new)

# Enable code style check

file(GLOB_RECURSE template_extension_src "${CMAKE_CURRENT_SOURCE_DIR}/new/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/new/*.hpp")
ov_add_clang_format_target(openvino_template_extension_clang FOR_SOURCES ${template_extension_src})
