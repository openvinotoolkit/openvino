# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_AUTO AND NOT ENABLE_MULTI)
    return()
endif()

set (TARGET_NAME "openvino_auto_plugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp)

# WA for issue 141313
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    ov_add_compiler_flags(-Wno-stringop-overflow)
endif()

if(ENABLE_AUTO AND ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)

    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  PSEUDO_PLUGIN_FOR "MULTI")
elseif(ENABLE_AUTO)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "AUTO"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
elseif(ENABLE_MULTI)
    ov_add_plugin(NAME ${TARGET_NAME}
                  DEVICE_NAME "MULTI"
                  PSEUDO_DEVICE
                  SOURCES ${SOURCES}
                  VERSION_DEFINES_FOR src/plugin.cpp)
endif()

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
