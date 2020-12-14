# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcTestUtils)

list(APPEND EXPORT_DEPENDENCIES
        inference_engine_lp_transformations
        commonTestUtils
        inference_engine
        inference_engine_legacy
        )

addIeTarget(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ADD_CPPLINT
    DEVELOPER_PACKAGE
    INCLUDES
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ADDITIONAL_SOURCE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    LINK_LIBRARIES
        PUBLIC
            ${EXPORT_DEPENDENCIES}
            inference_engine_transformations
    INCLUDES
        PUBLIC
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
    EXPORT_DEPENDENCIES
        ${EXPORT_DEPENDENCIES}
)

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

