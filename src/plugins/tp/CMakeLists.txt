# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_TP)
    return()
endif()

set(TARGET_NAME openvino_tp_plugin)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "TP"
              PSEUDO_DEVICE
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/version.cpp
              ADD_CLANG_FORMAT)

target_include_directories(${TARGET_NAME}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(ENABLE_TP_DEBUG_CAPS)
    target_compile_definitions(${TARGET_NAME} PRIVATE TP_DEBUG_CONFIG=1)
endif()

ov_build_target_faster(${TARGET_NAME}
    UNITY
)

find_package(Threads REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE
    Threads::Threads
    openvino::runtime::dev)  # transformations/utils/utils.hpp, the collective contract

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
