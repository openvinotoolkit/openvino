# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_HETERO)
    return()
endif()

set (TARGET_NAME "openvino_hetero_plugin")

# TODO (vurusovs) revert after finish of porting Hetero plugin to API 2.0
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "HETERO"
              PSEUDO_DEVICE
              SOURCES ${SOURCES} ${HEADERS}
              # TODO (vurusovs) revert after finish of porting Hetero plugin to API 2.0
              VERSION_DEFINES_FOR src/plugin.cpp
              ADD_CLANG_FORMAT)

ie_faster_build(${TARGET_NAME}
    UNITY
)

# TODO (vurusovs) revert after finish of porting Hetero plugin to API 2.0
target_include_directories(${TARGET_NAME} PRIVATE
    "${OpenVINO_SOURCE_DIR}/src/inference/src/dev"
    )

target_link_libraries(${TARGET_NAME} PRIVATE openvino::pugixml)

# must be called after all target_link_libraries
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
