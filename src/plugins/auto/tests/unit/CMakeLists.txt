# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_unit_tests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DMULTIUNITTEST)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/utils
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto/src
        LINK_LIBRARIES
            unit_test_utils
        ADD_CPPLINT
        DEPENDENCIES
            openvino_template_extension
            mock_engine
        LABELS
            OV UNIT MULTI AUTO
)

ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto/src/plugin.cpp ${TARGET_NAME})

ov_set_threading_interface_for(${TARGET_NAME})
