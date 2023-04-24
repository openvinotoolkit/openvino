# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_unit_tests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd5051)
endif()

if (NOT OPENVINO_ARCH_X86_64)
    set(EXCLUDED_SOURCE_PATHS_FOR_NON_X64 ${CMAKE_CURRENT_SOURCE_DIR}/jit_kernel_test.cpp
                                          ${CMAKE_CURRENT_SOURCE_DIR}/registers_pool.cpp
                                          ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_transformations/convert_to_interaction.cpp
                                          ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_transformations/snipptes_mark_skipped.cpp
                                          ${CMAKE_CURRENT_SOURCE_DIR}/ngraph_transformations/mul_add_to_fma.cpp
                                          ${CMAKE_CURRENT_SOURCE_DIR}/snippets_transformations
                                          ${CMAKE_CURRENT_SOURCE_DIR}/nodes/eltwise_node_test.cpp)
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
        EXCLUDED_SOURCE_PATHS
            ${EXCLUDED_SOURCE_PATHS_FOR_NON_X64}
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
            snippets_test_utils
        ADD_CPPLINT
        LABELS
            CPU
)

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

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
