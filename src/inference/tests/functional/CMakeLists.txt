# Copyright (C) 2018-2026 Intel Corporation
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
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/caching_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/get_supported_property_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/import_model.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/local_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/matmul_sr_tests.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_compiled_model_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_core_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_core_threading.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_extension_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_extension_test.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_infer_request_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_register_plugin_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_remote_context_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_remote_tensor.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_shared_object_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ov_variable_state_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/skip_tests_config.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/task_executor_tests.cpp
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
        CHECK_SOURCES_LISTED
)

ov_set_threading_interface_for(${TARGET_NAME})
