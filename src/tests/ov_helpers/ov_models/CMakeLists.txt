# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_models)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>"
                "$<INSTALL_INTERFACE:developer_package/include/ov_models>"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PUBLIC
                openvino::runtime
                openvino::reference
                interpreter_backend
                openvino::runtime::dev
                common_test_utils
        ADD_CLANG_FORMAT
        DEVELOPER_PACKAGE
            tests
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

# install

install(DIRECTORY "${PUBLIC_HEADERS_DIR}/"
        DESTINATION developer_package/include/ov_models
        COMPONENT developer_package EXCLUDE_FROM_ALL)
