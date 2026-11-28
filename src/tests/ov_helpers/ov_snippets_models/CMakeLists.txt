# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_snippets_models)

include(sources.cmake)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        SOURCES ${OV_SNIPPETS_MODELS_SOURCES}
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>"
                "$<BUILD_INTERFACE:${COMMON_TEST_UTILS_INCLUDES}>"
        LINK_LIBRARIES
            PUBLIC
                openvino::runtime::dev
                common_test_utils
                openvino::snippets
                ov_lpt_models
)

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH_HEADER "src/precomp.hpp"
)

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${PUBLIC_HEADERS_DIR}/")
