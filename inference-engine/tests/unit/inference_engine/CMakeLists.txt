# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ieUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            unitTestUtils
        ADD_CPPLINT
        DEPENDENCIES
            mock_engine
        LABELS
            IE
)

target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:inference_engine_ir_v7_reader,INTERFACE_INCLUDE_DIRECTORIES>)
