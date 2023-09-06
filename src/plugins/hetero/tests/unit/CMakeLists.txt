# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_unit_tests)

set(CI_BUILD_NUMBER "unittest")
set(HETERO_SOURCE_DIR ${OpenVINO_SOURCE_DIR}/src/plugins/hetero/src)

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${HETERO_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${HETERO_SOURCE_DIR}
        LINK_LIBRARIES
            unit_test_utils
            ngraphFunctions
        DEPENDENCIES
            mock_engine
            ngraphFunctions
        ADD_CLANG_FORMAT
        LABELS
            HETERO
)
ov_add_version_defines(${HETERO_SOURCE_DIR}/plugin.cpp ${TARGET_NAME})

set_ie_threading_interface_for(${TARGET_NAME})
