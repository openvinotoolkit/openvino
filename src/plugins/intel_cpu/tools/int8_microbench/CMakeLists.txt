# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_int8_microbench)

add_executable(${TARGET_NAME}
    main.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/jit_int8_conv_kernel.cpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/nodes/kernels/aarch64/brgemm_int8_kernel.cpp)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src
    ${OpenVINO_SOURCE_DIR}/src/core/include
    ${OpenVINO_SOURCE_DIR}/src/inference/include
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
