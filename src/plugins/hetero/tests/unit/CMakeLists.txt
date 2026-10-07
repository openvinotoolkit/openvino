# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_unit_tests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_hetero_plugin_obj>)
endif()

include(sources.cmake)

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        SOURCES
            ${OV_HETERO_UNIT_TESTS_SRC}
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_hetero_plugin,SOURCE_DIR>/src
            ${CMAKE_CURRENT_SOURCE_DIR}
        OBJECT_FILES
            ${OBJ_LIB}
        LINK_LIBRARIES
            unit_test_utils
        DEPENDENCIES
            mock_engine
        ADD_CLANG_FORMAT
        CHECK_SOURCES_LISTED
        LABELS
            OV UNIT HETERO
)
