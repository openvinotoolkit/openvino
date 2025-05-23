# Copyright (C) 2018-2025 Intel Corporation
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
else()
    list(APPEND EXCLUDED_SOURCE_PATHS_FOR_UNIT_TEST
      ${CMAKE_CURRENT_SOURCE_DIR}/stub.cpp)
endif()

if (ENABLE_MLAS_FOR_CPU)
    set(MLAS_LIBRARY "mlas")
endif()

if (ENABLE_SHL_FOR_CPU)
    set(SHL_LIBRARY "shl")
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
            ov_snippets_models
            snippets_test_utils
            ${MLAS_LIBRARY}
            ${SHL_LIBRARY}
        ADD_CPPLINT
        LABELS
            OV UNIT CPU
)


if (ENABLE_SNIPPETS_LIBXSMM_TPP)
    add_definitions(-DSNIPPETS_LIBXSMM_TPP -DLIBXSMM_DEFAULT_CONFIG)
    target_compile_definitions(xsmm PRIVATE __BLAS=0)
    target_link_libraries(${TARGET_NAME} PRIVATE xsmm)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>)
endif()

if(X86_64)
    if(ENABLE_AVX2)
        ov_avx2_optimization_flags(avx2_flags)
        list(APPEND compile_flags ${avx2_flags})
        list(APPEND definitions HAVE_AVX2)
    endif()
    if(ENABLE_AVX512F)
        ov_avx512_optimization_flags(avx512_flags)
        list(APPEND compile_flags ${avx512_flags})
        list(APPEND definitions HAVE_AVX512F)
    endif()

    target_compile_options(${TARGET_NAME} PRIVATE "${compile_flags}")
    target_compile_definitions(${TARGET_NAME} PRIVATE ${definitions})
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
