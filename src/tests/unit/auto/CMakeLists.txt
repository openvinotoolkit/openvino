# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieMultiPluginUnitTests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DMULTIUNITTEST)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS ${OpenVINO_SOURCE_DIR}/src/plugins/auto ${OpenVINO_SOURCE_DIR}/src/plugins/auto/utils
        INCLUDES
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            ngraphFunctions
            openvino::runtime
            openvino::runtime::dev
            unitTestUtils
        ADD_CPPLINT
        DEPENDENCIES
            template_extension
            mock_engine
            ngraphFunctions
        LABELS
            Multi
            Auto
)
ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto/plugin.cpp ${TARGET_NAME})

set_ie_threading_interface_for(${TARGET_NAME})
