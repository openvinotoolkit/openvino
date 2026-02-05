# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME test_lib)

add_library(${TARGET_NAME} STATIC test_lib.cpp)
target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:openvino::lib_load_unload,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME test_lib)

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET test_lib
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")

