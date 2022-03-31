# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "mock_bde_plugin")

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "BDE"
              SOURCES ${SOURCES} ${HEADERS}
              SKIP_INSTALL
              SKIP_REGISTRATION
              DEFAULT_CONFIG
                  PROXY_NAME:MOCK;PROXY_PRIORITY:2
              ADD_CLANG_FORMAT)

target_link_libraries(${TARGET_NAME} PRIVATE mock_plugin_shared)

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT tests EXCLUDE_FROM_ALL)
