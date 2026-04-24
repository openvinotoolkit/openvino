# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_INT8_AARCH64_MICROBENCH)
    return()
endif()

set(TARGET_NAME ov_cpu_int8_microbench)

add_executable(${TARGET_NAME}
    main.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8_1x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8_1x4_neon_mla.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8_1x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8_2x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8_4x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_1x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_2x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_4x16_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_4x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_4x4_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_4x8_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_8x12_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qsi8_qsi8p_8x8_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8_1x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8_1x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8_2x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8_4x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_1x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_2x8_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_4x16_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_4x4_neon_dotprod.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_4x4_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_4x8_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_8x12_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_brgemm_s32_qu8_qsi8p_8x8_neon_i8mm.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_kernels/ov_dot_s32_qsi8_qsi8_1x1_neon.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_int8_kernel.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/precision_support.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
    ${OpenVINO_SOURCE_DIR}/src/core/include
    ${OpenVINO_SOURCE_DIR}/src/inference/include
    ${OpenVINO_SOURCE_DIR}/src/inference/dev_api
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/src
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/thirdparty/onednn/third_party)

target_link_libraries(${TARGET_NAME} PRIVATE dnnl openvino::runtime)

if(OV_CPU_WITH_ACL)
    target_link_libraries(${TARGET_NAME} PRIVATE arm_compute::arm_compute)
    target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:arm_compute::arm_compute,SOURCE_DIR>)
endif()

if(ENABLE_KLEIDIAI_FOR_CPU)
    target_link_libraries(${TARGET_NAME} PRIVATE kleidiai)
    target_include_directories(${TARGET_NAME} PRIVATE $<TARGET_PROPERTY:kleidiai,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${TARGET_NAME} PRIVATE OV_CPU_WITH_KLEIDIAI)
endif()
