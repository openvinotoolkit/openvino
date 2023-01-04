# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
if(NOT ENABLE_AUTO AND NOT ENABLE_MULTI)
    return()
endif()



set (TARGET_NAME "openvino_auto_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

if(ENABLE_AUTO AND ENABLE_MULTI)
    ie_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR plugin.cpp)

    ie_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  PSEUDO_PLUGIN_FOR "MULTI")
elseif(ENABLE_AUTO)
    ie_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR plugin.cpp)
elseif(ENABLE_MULTI)
    ie_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR plugin.cpp)
endif()

set_ie_threading_interface_for(${TARGET_NAME})

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
