# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_batch_func_tests)

set(SHARED_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/tests/functional/plugin/shared/include")
ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${SHARED_HEADERS_DIR}
        LINK_LIBRARIES
            funcSharedTests
        DEPENDENCIES
            openvino_auto_batch_plugin
        ADD_CPPLINT
        LABELS
            OV AUTO_BATCH
)

ov_set_threading_interface_for(${TARGET_NAME})
