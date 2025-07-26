cmake_minimum_required(VERSION 3.21)
project(ov_cpu_func_tests)

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME ov_cpu_func_tests)

if(SUGGEST_OVERRIDE_SUPPORTED)
    # xbyak compilation fails
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

# To allow test classes call `mayiuse(gv)`
if(RISCV64)
    set(CPU_ISA_TRAITS_RV64 $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/riscv64/cpu_isa_traits.hpp
                            $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/riscv64/cpu_isa_traits.cpp)
endif()

add_library(cpuUtils STATIC
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.hpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/precision_support.h
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/precision_support.cpp
    ${CPU_ISA_TRAITS_RV64})
set(CPU_UTILS_LINK_LIBRARIES openvino::runtime::dev)
set(CPU_UTILS_INCLUDE_PATHS)
if(OV_CPU_WITH_ACL)
    list(APPEND CPU_UTILS_LINK_LIBRARIES arm_compute::arm_compute)
    list(APPEND CPU_UTILS_INCLUDE_PATHS $<TARGET_PROPERTY:arm_compute::arm_compute,SOURCE_DIR>)
endif()
if(OV_CPU_WITH_DNNL)
    list(APPEND CPU_UTILS_LINK_LIBRARIES dnnl)
    list(APPEND CPU_UTILS_INCLUDE_PATHS
        $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/src
        $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/third_party)
endif()
if(RISCV64)
    list(APPEND CPU_UTILS_LINK_LIBRARIES xbyak_riscv)
endif()
target_link_libraries(cpuUtils PRIVATE ${CPU_UTILS_LINK_LIBRARIES})
target_include_directories(cpuUtils PRIVATE ${CPU_UTILS_INCLUDE_PATHS})

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin openvino_template_extension)
set(LINK_LIBRARIES funcSharedTests cpuUtils openvino::snippets ov_snippets_models)

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
endif()

# Initialize the global property
set_property(GLOBAL PROPERTY CPU_FUNC_TESTS_SRC "")

# Add subdirectories to collect source files
# Architecture-specific inclusion is now handled in each subdirectory's CMakeLists.txt
add_subdirectory(shared_tests_instances)
add_subdirectory(utils)

# Handle custom tests with special ARM logic
if(ARM OR AARCH64)
    # For ARM, only include specific subdirectories (temporary disable as per original logic)
    add_subdirectory(custom/behavior)
    add_subdirectory(custom/single_layer_tests/classes)
    add_subdirectory(custom/single_layer_tests/instances/common)
    add_subdirectory(custom/single_layer_tests/instances/arm)
    add_subdirectory(custom/subgraph_tests/include)
    add_subdirectory(custom/subgraph_tests/src/classes)
    add_subdirectory(custom/subgraph_tests/src/common)
    add_subdirectory(custom/subgraph_tests/src/arm)
    if(ENABLE_OV_ONNX_FRONTEND)
        add_subdirectory(custom/extension)
    endif()
else()
    # For non-ARM platforms, include all custom tests
    add_subdirectory(custom)
endif()

# Get all collected source files
get_property(ALL_CPU_FUNC_TESTS_SRC GLOBAL PROPERTY CPU_FUNC_TESTS_SRC)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        OBJECT_FILES ${ALL_CPU_FUNC_TESTS_SRC}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS OV CPU
)

ov_set_threading_interface_for(${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)

include(cmake/specific_tests.cmake)
include(cmake/target_per_test.cmake)

# exclude 'ov_cpu_func_tests' from target 'all' if specific tests path was specified
if(DEFINED ENABLE_CPU_SPECIFIC_TESTS_PATH)
    set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
endif()