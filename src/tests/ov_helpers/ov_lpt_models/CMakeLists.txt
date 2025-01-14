# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_lpt_models)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PRIVATE
                openvino::runtime::dev
                common_test_utils
        ADD_CPPLINT
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH PRIVATE "src/precomp.hpp"
)

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
