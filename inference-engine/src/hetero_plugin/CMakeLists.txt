# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "HeteroPlugin")

if(ENABLE_LTO)
    ie_enable_lto()
endif()

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "HETERO"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR hetero_plugin.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ade pugixml)
