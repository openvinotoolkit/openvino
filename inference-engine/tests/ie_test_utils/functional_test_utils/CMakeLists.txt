# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcTestUtils)

list(APPEND EXPORT_DEPENDENCIES commonTestUtils inference_engine inference_engine_legacy)

list(APPEND EXPORT_DEPENDENCIES inference_engine_lp_transformations)

addIeTarget(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CPPLINT
    DEVELOPER_PACKAGE
    LINK_LIBRARIES ngraphFunctions
    EXPORT_DEPENDENCIES ${EXPORT_DEPENDENCIES}
)

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "precomp.hpp"
)

target_include_directories(${TARGET_NAME} PUBLIC
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(${TARGET_NAME} PUBLIC ${EXPORT_DEPENDENCIES} inference_engine_transformations)
