# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ngraphFunctions)

list(APPEND EXPORT_DEPENDENCIES
        ${NGRAPH_LIBRARIES}
        ngraph_backend
        interpreter_backend
        inference_engine_lp_transformations
        )

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            funcTestUtils
        ADD_CPPLINT
        DEVELOPER_PACKAGE
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR})

target_link_libraries(${TARGET_NAME} PUBLIC ${EXPORT_DEPENDENCIES})
