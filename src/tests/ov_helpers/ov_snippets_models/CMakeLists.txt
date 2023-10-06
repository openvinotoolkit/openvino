# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_models)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SNIPPETS_INCLUDES "$<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/include")
set(COMMON_TEST_UTILS_INCLUDES "$<TARGET_PROPERTY:common_test_utils,INTERFACE_INCLUDE_DIRECTORIES>")

ov_add_target(
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
                common_test_utils
                openvino::snippets
                ov_lpt_models
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)
