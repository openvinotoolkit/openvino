# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_unit_tests_vectorized)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd5051)
endif()

# only one of these files is ever built - the other is unsupported / redundant on this arch
set(CPU_UNIT_TESTS_VECTORIZED_EXCLUDE_FILES)
if(X86_64)
    set(CPU_UNIT_TESTS_VECTORIZED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/paged_attn_cache_rotation.cpp)
    list(APPEND CPU_UNIT_TESTS_VECTORIZED_EXCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/stub.cpp)
else()
    set(CPU_UNIT_TESTS_VECTORIZED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/stub.cpp)
    list(APPEND CPU_UNIT_TESTS_VECTORIZED_EXCLUDE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/paged_attn_cache_rotation.cpp)
endif()

if (ENABLE_MLAS_FOR_CPU)
    set(MLAS_LIBRARY "mlas")
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES ${CPU_UNIT_TESTS_VECTORIZED_SRCS}
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes
                $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
            PRIVATE
                $<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/include
        OBJECT_FILES
            ${OBJ_LIB}
        LINK_LIBRARIES
            gtest
            gtest_main
            dnnl
            gmock
            openvino_runtime_s
            openvino_xml_util
            unit_test_utils
            ov_snippets_models
            snippets_test_utils
            ${MLAS_LIBRARY}
        LABELS
            OV UNIT CPU
        CHECK_SOURCES_LISTED
        CHECK_SOURCES_EXCLUDE_FILES
            ${CPU_UNIT_TESTS_VECTORIZED_EXCLUDE_FILES}
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

# explicitly set include directories for gtest before the dnnl include directories as dnnl third_party dir also contains
# gtest. This is necessary to ensure a proper order of include directories in the compilation command line
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:gtest_main,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:gmock,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>)
