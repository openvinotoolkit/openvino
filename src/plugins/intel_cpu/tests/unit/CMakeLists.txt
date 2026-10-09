# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(vectorized)

set(TARGET_NAME ov_cpu_unit_tests)

if(BUILD_SHARED_LIBS)
    set (OBJ_LIB $<TARGET_OBJECTS:openvino_intel_cpu_plugin_obj>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd5051)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

if(ENABLE_MLAS_FOR_CPU)
    set(MLAS_LIBRARY "mlas")
endif()

if (ENABLE_KLEIDIAI_FOR_CPU)
    set(KLEIDIAI_LIBRARY "kleidiai")
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${CPU_UNIT_TESTS_SRCS}
        INCLUDES
            PUBLIC
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn
                $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/src
                $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:openvino::reference,SOURCE_DIR>/include
            PRIVATE
                $<TARGET_PROPERTY:openvino::snippets,SOURCE_DIR>/include
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
            openvino_xml_util
            ov_snippets_models
            snippets_test_utils
            ${MLAS_LIBRARY}
            ${KLEIDIAI_LIBRARY}
        LABELS
            OV UNIT CPU
        CHECK_SOURCES_LISTED
        CHECK_SOURCES_EXCLUDE_FILES
            ${CPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES}
        CHECK_SOURCES_EXCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_SOURCE_DIR}/vectorized
)

if (ENABLE_SNIPPETS_LIBXSMM_TPP)
    add_definitions(-DSNIPPETS_LIBXSMM_TPP -DLIBXSMM_DEFAULT_CONFIG)
    target_compile_definitions(xsmm PRIVATE __BLAS=0)
    target_link_libraries(${TARGET_NAME} PRIVATE xsmm)
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:xsmm,INCLUDE_DIRECTORIES>)
endif()


# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (WIN32)
    # Prevents defining min/max as macros
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
endif()

ov_build_target_faster(${TARGET_NAME}
    UNITY
    PCH
)

# unity groups are assigned in sources.cmake
if(ENABLE_UNITY_BUILD)
    set_target_properties(${TARGET_NAME} PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
endif()

# explicitly set include directories for gtest before the dnnl include directories as dnnl third_party dir also contains
# gtest. This is necessary to ensure a proper order of include directories in the compilation command line
target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:gtest_main,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:gmock,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,SOURCE_DIR>)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)
