# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME openvino_protobuf_shutdown)

add_library(${TARGET_NAME} STATIC shutdown_protobuf.cpp)

add_library(openvino::protobuf_shutdown ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME protobuf_shutdown)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>>)
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
target_compile_features(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_FEATURES>)
