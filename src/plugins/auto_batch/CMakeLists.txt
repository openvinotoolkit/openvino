# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_AUTO_BATCH)
    return()
endif()

set(TARGET_NAME "openvino_auto_batch_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "BATCH"
              PSEUDO_DEVICE
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/plugin.cpp ADD_CLANG_FORMAT)

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
