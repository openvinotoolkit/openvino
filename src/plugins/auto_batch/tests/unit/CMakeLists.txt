# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_auto_batch_unit_tests)

set(CI_BUILD_NUMBER "unittest")
add_definitions(-DAUTOBATCH_UNITTEST)

set(SHARED_HEADERS_DIR "${OpenVINO_SOURCE_DIR}/src/tests/ie_test_util")

include(sources.cmake)

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        SOURCES
            ${OV_AUTO_BATCH_UNIT_TESTS_SRC}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src
            ${SHARED_HEADERS_DIR}
        LINK_LIBRARIES
            unit_test_utils
        DEPENDENCIES
            mock_engine
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT AUTO_BATCH
)

ov_add_version_defines(${OpenVINO_SOURCE_DIR}/src/plugins/auto_batch/src/plugin.cpp ${TARGET_NAME})

ov_set_threading_interface_for(${TARGET_NAME})
