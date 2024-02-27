# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_unit_tests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd5051)
endif()

if(NOT (ARM OR AARCH64))
    list(APPEND EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST
      ${CMAKE_CURRENT_SOURCE_DIR}/transformations/arm)
endif()

if(NOT X86_64)
    list(APPEND EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST
      ${CMAKE_CURRENT_SOURCE_DIR}/jit_kernel_test.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/registers_pool.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/transformations/x64
      ${CMAKE_CURRENT_SOURCE_DIR}/snippets_transformations
      ${CMAKE_CURRENT_SOURCE_DIR}/nodes/eltwise_node_test.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/brgemm_executor_test.cpp)
endif()

if (NOT ENABLE_MLAS_FOR_CPU)
    list(APPEND EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST ${CMAKE_CURRENT_SOURCE_DIR}/gemm_api_test.cpp)
else()
    set(MLAS_LIBRARY "mlas")
endif()

ov_add_test_target(
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
                $<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/include
        EXCLUDED_SOURCE_PATHS
            ${EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST}
        OBJECT_FILES
            ${OBJ_LIB}
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            dnnl
            openvino::shape_inference
            openvino_runtime_s
            unit_test_utils
            ov_snippets_models
            snippets_test_utils
            ${MLAS_LIBRARY}
        ADD_CPPLINT
        LABELS
            OV UNIT CPU
)

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

if (WIN32)
    # Prevents defining min/max as macros
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
endif()

ov_build_target_faster(${TARGET_NAME}
    UNITY
)

function(group_source_file GROUP_NAME GROUP_DIR)
    file(GLOB GROUP_FILES  ${GROUP_DIR}/*.cpp)
    foreach(file ${GROUP_FILES})
        set_source_files_properties(${file} PROPERTIES UNITY_GROUP ${GROUP_NAME})
    endforeach()
endfunction()

if(ENABLE_FASTER_BUILD) 
    set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
    group_source_file(unit_src_nodes ${CMAKE_CURRENT_SOURCE_DIR}/nodes)
    group_source_file(unit_src_snippets_transformations ${CMAKE_CURRENT_SOURCE_DIR}/snippets_transformations)
    group_source_file(unit_src_transformations ${CMAKE_CURRENT_SOURCE_DIR}/transformations)
    group_source_file(unit_src_custom_shape_infer ${CMAKE_CURRENT_SOURCE_DIR}/shape_inference_test/custom_shape_infer)
endif()

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/common
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/cpu
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/include)
