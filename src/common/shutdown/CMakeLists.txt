# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_shutdown)

add_library(${TARGET_NAME} STATIC shutdown.cpp)

add_library(openvino::shutdown ALIAS ${TARGET_NAME})
target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME shutdown)
ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

# install & export
ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
ov_developer_package_export_targets(TARGET openvino::shutdown
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
