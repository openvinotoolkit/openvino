# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME lptNgraphFunctions)

list(APPEND EXPORT_DEPENDENCIES
        ngraphFunctions
        inference_engine_lp_transformations
        inference_engine_legacy
        )

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        INCLUDES
            PUBLIC
                ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PRIVATE
                ${EXPORT_DEPENDENCIES}
        ADD_CPPLINT
        DEPENDENCIES
            ngraphFunctions
        DEVELOPER_PACKAGE
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)
