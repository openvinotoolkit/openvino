# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/src
                $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
            PRIVATE
                $<TARGET_PROPERTY:inference_engine_snippets,SOURCE_DIR>/include
        OBJECT_FILES
            ${OBJ_LIB}
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            dnnl
            inference_engine_transformations
            inference_engine_lp_transformations
            ov_shape_inference
            inference_engine_s
            unitTestUtils
            ngraphFunctions
            snippetsNgraphFunctions
        ADD_CPPLINT
        LABELS
            CPU
)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

if (WIN32)
    # Prevents defining min/max as macros
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
endif()

ie_faster_build(${TARGET_NAME}
    UNITY
)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/common
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/cpu
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/include)
