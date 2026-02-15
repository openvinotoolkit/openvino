# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_hetero_unit_tests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_hetero_plugin_obj>)
endif()

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
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
        LABELS
            OV UNIT HETERO
)
