# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

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
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom/extension ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/onnx)
endif()

if(NOT RISCV64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64)
elseif(NOT OV_CPU_WITH_SHL)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64/shl)
endif()

if(NOT (X86 OR X86_64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64)
endif()

if(NOT (ARM OR AARCH64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/snippets/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/aarch64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm)
else()
    # temporary disable all custom tests for ARM except explicitly enabled ones
    list(APPEND EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests)
endif()

# Reset global accumulators before diving into subdirectories
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "")
set_property(GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS "")

# Natural shallow recursion — each subdir lists only its own files and recurses to its children
add_subdirectory(utils)
add_subdirectory(shared_tests_instances)
add_subdirectory(custom)

# Retrieve aggregated list
get_property(CPU_FUNC_TEST_SRCS GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)

# Keep IDE structure
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${CPU_FUNC_TEST_SRCS})

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS OV CPU
)

# Attach explicit sources to the target directly
target_sources(${TARGET_NAME} PRIVATE ${CPU_FUNC_TEST_SRCS})

ov_set_threading_interface_for(${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)

include(cmake/specific_tests.cmake)
include(cmake/target_per_test.cmake)

# exclude 'ov_cpu_func_tests' from target 'all' if specific tests path was specified
if(DEFINED ENABLE_CPU_SPECIFIC_TESTS_PATH)
    set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
