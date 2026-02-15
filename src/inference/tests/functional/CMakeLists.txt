# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_inference_functional_tests)

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

set(DEPENDENCIES
    mock_engine
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

if(ENABLE_PROXY)
    list(APPEND COMPILE_DEFINITIONS PROXY_PLUGIN_ENABLED)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ${DEPENDENCIES}
        LINK_LIBRARIES
            gmock
            func_test_utils
        DEFINES
            ${COMPILE_DEFINITIONS}
        INCLUDES
            $<TARGET_PROPERTY:openvino_runtime_obj,SOURCE_DIR>/src
            $<$<TARGET_EXISTS:openvino_proxy_plugin_obj>:$<TARGET_PROPERTY:openvino_proxy_plugin_obj,INTERFACE_INCLUDE_DIRECTORIES>>
            ${CMAKE_CURRENT_SOURCE_DIR}
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT RUNTIME
)

ov_set_threading_interface_for(${TARGET_NAME})
