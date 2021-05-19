# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "AutoPlugin")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "AUTO"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR auto_plugin.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES} inference_engine inference_engine_transformations)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
