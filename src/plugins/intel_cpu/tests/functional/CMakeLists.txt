# Copyright (C) 2018-2026 Intel Corporation
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
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom/extension ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/onnx)
endif()

if(NOT RISCV64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64)
else()
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/riscv64/shl)
endif()

if(NOT (X86 OR X86_64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64)
endif()

# Benchmark sources live in dedicated directories and are built into a separate
# ov_cpu_benchmark_tests binary — exclude them from the main ov_cpu_func_tests.
list(APPEND EXCLUDED_SOURCE_PATHS
     ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/benchmark
     ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/benchmark)

if(NOT (ARM OR AARCH64))
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/aarch64
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm)
else()
    # temporary disable all custom tests for ARM
    list(APPEND EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests)
    # except ones which already enabled
    file(GLOB_RECURSE TMP_LIST_OF_TEST_CLASSES          ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_COMMON_TEST_INSTANCES ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/common/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_ARM_TEST_INSTANCES    ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/arm/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_ARM_SUBGRAPH_TESTS    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_COMMON_SUBGRAPH_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_SUBGRAPH_TEST_CLASSES ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/*.*)

    list(APPEND TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS
        ${TMP_LIST_OF_TEST_CLASSES} ${TMP_LIST_OF_COMMON_TEST_INSTANCES} ${TMP_LIST_OF_ARM_TEST_INSTANCES} ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS} ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS} ${TMP_LIST_OF_SUBGRAPH_TEST_CLASSES})
    set(TMP_EXPLICITLY_ENABLED_TESTS "${TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS}")
endif()

if(NOT X86_64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/low_precision_transformations/x64)
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        OBJECT_FILES ${TMP_EXPLICITLY_ENABLED_TESTS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
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

# ---------------------------------------------------------------------------
# ov_cpu_benchmark_tests: separate binary bundling benchmark-style tests.
# Kept outside ov_cpu_func_tests — their runtime is dominated by measurement
# loops, not correctness verification.
# Sources live under custom/{subgraph,single_layer}_tests/benchmark/.
# benchmark instances can reuse ov_cpu_func_tests test classes or define their own.
# ---------------------------------------------------------------------------
set(BENCH_TARGET_NAME ov_cpu_benchmark_tests)

# Cross-platform benchmark sources (test class definitions).
set(BENCH_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/benchmark/classes/concat_sdp_kv_bench.cpp)

# Architecture-specific benchmark instance sources.
if(X86_64)
    list(APPEND BENCH_TEST_SOURCES
         ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/benchmark/x64/concat_sdp_kv_bench.cpp)
endif()

set(BENCH_REQUIRED_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp)
if(X86_64)
    list(APPEND BENCH_REQUIRED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
    list(APPEND BENCH_REQUIRED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
elseif(RISCV64)
    list(APPEND BENCH_REQUIRED_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
endif()

# ROOT points at the benchmark class dir (kept narrow so recursive globbing
# doesn't pick up unrelated sources); EXCLUDED_SOURCE_PATHS drops it entirely
# and we add back only the explicit sources via OBJECT_FILES.
ov_add_test_target(
    NAME ${BENCH_TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/benchmark
    INCLUDES ${INCLUDES}
    EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/benchmark
    OBJECT_FILES ${BENCH_REQUIRED_FILES} ${BENCH_TEST_SOURCES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    # Separate install component — CI's 'tests' install step won't try to copy
    # this binary if it wasn't built (EXCLUDE_FROM_ALL below).
    COMPONENT benchmark_tests
    LABELS OV CPU BENCHMARK)

ov_set_threading_interface_for(${BENCH_TARGET_NAME})
set_target_properties(${BENCH_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
