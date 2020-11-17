# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcTestUtils)

list(APPEND EXPORT_DEPENDENCIES
        inference_engine_lp_transformations
        commonTestUtils
        inference_engine
        inference_engine_legacy
        ngraphFunctions
        inference_engine_transformations
        )

addIeTarget(
    NAME ${TARGET_NAME}
    TYPE STATIC
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    ADD_CPPLINT
    DEVELOPER_PACKAGE
    LINK_LIBRARIES
        PUBLIC
            ${EXPORT_DEPENDENCIES}
    INCLUDES
        PUBLIC
            $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
#    LINK_FLAGS
#        PUBLIC
#        IR_SERIALIZATION_MODELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ir_serialization/models/"
    DEPENDENCIES
        ${EXPORT_DEPENDENCIES}
    EXPORT_DEPENDENCIES ${EXPORT_DEPENDENCIES}
)

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "precomp.hpp"
)

# ir serialization functional tests variables
target_compile_definitions(${TARGET_NAME} INTERFACE IR_SERIALIZATION_MODELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ir_serialization_models/")
