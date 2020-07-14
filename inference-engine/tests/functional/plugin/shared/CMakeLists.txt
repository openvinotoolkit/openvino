# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcSharedTests)

list(APPEND EXPORT_DEPENDENCIES
        funcTestUtils
        ngraphFunctions
        )

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        DEVELOPER_PACKAGE
        DEPENDENCIES
            HeteroPlugin
            MultiDevicePlugin
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

if (TARGET MKLDNNPlugin)
    add_dependencies(${TARGET_NAME} MKLDNNPlugin)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
        ${PUBLIC_HEADERS_DIR}
        ${IE_MAIN_SOURCE_DIR}/src/vpu/graph_transformer/include
        )

target_link_libraries(${TARGET_NAME}
        PUBLIC
        ${EXPORT_DEPENDENCIES}

        PRIVATE
        inference_engine_transformations
        )