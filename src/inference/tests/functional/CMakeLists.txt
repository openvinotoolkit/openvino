# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    ie_add_compiler_flags(-Wno-suggest-override)
endif()

set(TARGET_NAME ov_inference_functional_tests)

set(DEPENDENCIES
    mock_engine
    template_extension
    openvino_template_extension
)

set(COMPILE_DEFINITIONS "")

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
    list(APPEND COMPILE_DEFINITIONS ENABLE_OV_IR_FRONTEND)
endif()

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_HETERO)
endif()

if(ENABLE_AUTO AND ENABLE_MULTI)
    list(APPEND DEPENDENCIES openvino_auto_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO)
endif()

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
    list(APPEND COMPILE_DEFINITIONS ENABLE_AUTO_BATCH)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            gmock
            funcTestUtils
        INCLUDES
            $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src
        ADD_CLANG_FORMAT
        LABELS
            OV
)

add_compile_definitions(${TARGET_NAME} ${COMPILE_DEFINITIONS})

