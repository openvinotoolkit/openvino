# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "MultiDevicePlugin")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "MULTI"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR multi_device.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine)

set_ie_threading_interface_for(${TARGET_NAME})
