# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ngraphFunctions)

list(APPEND EXPORT_DEPENDENCIES
        ${NGRAPH_LIBRARIES}
        ngraph_backend
        interpreter_backend
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
            # this import as result of including nGraph function building for tests
            inference_engine_transformations
        ADD_CPPLINT
        DEVELOPER_PACKAGE
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR})

target_link_libraries(${TARGET_NAME} PUBLIC ${EXPORT_DEPENDENCIES})
