# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME snippetsNgraphFunctions)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SNIPPETS_INCLUDES "$<TARGET_PROPERTY:inference_engine_snippets,SOURCE_DIR>/include")
set(COMMON_TEST_UTILS_INCLUDES "$<TARGET_PROPERTY:commonTestUtils,INTERFACE_INCLUDE_DIRECTORIES>")
addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        INCLUDES
            PUBLIC
                ${PUBLIC_HEADERS_DIR}
                ${COMMON_TEST_UTILS_INCLUDES}
            PRIVATE
                ${SNIPPETS_INCLUDES}

        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PUBLIC
                openvino::runtime::dev
                commonTestUtils
                inference_engine_snippets
                lptNgraphFunctions
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)
