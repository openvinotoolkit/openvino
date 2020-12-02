# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "MultiDevicePlugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "MULTI"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR multi_device_plugin.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

set_ie_threading_interface_for(${TARGET_NAME})

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

ie_add_manifest(TARGET_NAME ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
