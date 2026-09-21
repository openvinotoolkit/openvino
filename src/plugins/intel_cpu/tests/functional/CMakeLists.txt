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
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/arm_isa_support.h
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/arm_isa_support.cpp
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
set(LINK_LIBRARIES funcSharedTests cpuUtils openvino::snippets ov_snippets_models openvino::reference)

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

set(CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES)
set(CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_DIRECTORIES)

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_ONNX_SRCS})
else()
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${CPU_FUNC_TESTS_ONNX_SRCS})
endif()

if(RISCV64)
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_RISCV64_SRCS})
else()
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${CPU_FUNC_TESTS_RISCV64_SRCS})
endif()

if(X86 OR X86_64)
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_X86_SRCS})
else()
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${CPU_FUNC_TESTS_X86_SRCS})
endif()

if(X86_64)
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_X64_SRCS})
else()
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${CPU_FUNC_TESTS_X64_SRCS})
endif()

if(ARM OR AARCH64)
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_ARM_SRCS})
    # custom tests are temporarily disabled for ARM, except ones which are already enabled
    list(APPEND TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS
        ${TMP_LIST_OF_TEST_CLASSES} ${TMP_LIST_OF_COMMON_TEST_INSTANCES} ${TMP_LIST_OF_ARM_TEST_INSTANCES}
        ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS} ${TMP_LIST_OF_COMMON_SUBGRAPH_TESTS} ${TMP_LIST_OF_SUBGRAPH_TEST_CLASSES})
    set(TMP_EXPLICITLY_ENABLED_TESTS "${TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS}")
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests)
else()
    list(APPEND CPU_FUNC_TESTS_SRCS ${CPU_FUNC_TESTS_CUSTOM_SRCS})
    list(APPEND CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES
        ${CPU_FUNC_TESTS_ARM_SRCS} ${TMP_LIST_OF_ARM_TEST_INSTANCES} ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS})
endif()

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES ${CPU_FUNC_TESTS_SRCS}
        INCLUDES ${INCLUDES}
        OBJECT_FILES ${TMP_EXPLICITLY_ENABLED_TESTS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        LABELS OV CPU
        CHECK_SOURCES_LISTED
        CHECK_SOURCES_EXCLUDE_FILES
            ${CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_FILES}
        CHECK_SOURCES_EXCLUDE_DIRECTORIES
            ${CPU_FUNC_TESTS_CHECK_SOURCES_EXCLUDE_DIRECTORIES}
)

ov_set_threading_interface_for(${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)

include(cmake/specific_tests.cmake)
include(cmake/target_per_test.cmake)

# exclude 'ov_cpu_func_tests' from target 'all' if specific tests path was specified
if(DEFINED ENABLE_CPU_SPECIFIC_TESTS_PATH)
    set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
