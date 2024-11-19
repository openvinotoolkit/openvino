# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_unit_tests_vectorized)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd5051)
endif()

if(NOT X86_64)
    list(APPEND EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST
      ${CMAKE_CURRENT_SOURCE_DIR}/paged_attn_cache_rotation.cpp)
endif()


ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes
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
            dnnl
            gmock
            openvino_runtime_s
            unit_test_utils
            snippets_test_utils
        ADD_CPPLINT
        LABELS
            OV UNIT CPU
)


if (X86_64)
    ov_avx2_optimization_flags(avx2_flags)
    ov_avx512_optimization_flags(avx512_flags)

    target_compile_options(${TARGET_NAME} PRIVATE "${avx2_flags};${avx512_flags}")
    target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_AVX2 HAVE_AVX512F)
endif()


if (WIN32)
    # Prevents defining min/max as macros
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
endif()

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/common
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/src/cpu
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>/include)
