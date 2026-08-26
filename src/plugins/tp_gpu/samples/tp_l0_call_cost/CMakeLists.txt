# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Microbenchmark: host-side cost of the Level Zero entry points the TP_GPU
# collective uses while recording and submitting command lists.

set(TARGET_NAME tp_l0_call_cost)

if(NOT TARGET LevelZero::LevelZero)
    message(STATUS "tp_l0_call_cost: Level Zero not available; sample disabled")
    return()
endif()

find_program(OCLOC_EXE ocloc)
if(NOT OCLOC_EXE)
    message(STATUS "tp_l0_call_cost: 'ocloc' (Intel offline compiler) not found; sample disabled")
    return()
endif()

# Timing AppendLaunchKernel needs a real kernel, and the plugin's own source
# is used so the measurement matches what the collective actually records.
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/allreduce_sum.cl)
set(KERNEL_SPV ${CMAKE_CURRENT_BINARY_DIR}/allreduce_sum.spv)
set(KERNEL_HDR ${CMAKE_CURRENT_BINARY_DIR}/allreduce_sum_spv.h)

add_custom_command(
    OUTPUT  ${KERNEL_SPV}
    COMMAND ${OCLOC_EXE} -file ${KERNEL_SRC}
                         -device dg2
                         -spv_only
                         -output allreduce_sum
                         -output_no_suffix
                         -out_dir ${CMAKE_CURRENT_BINARY_DIR}
                         -options "-cl-std=CL2.0"
                         -q
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling ${KERNEL_SRC} -> SPIR-V (ocloc)"
    VERBATIM)

add_custom_command(
    OUTPUT  ${KERNEL_HDR}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_FILE=${KERNEL_SPV}
            -DHDR_FILE=${KERNEL_HDR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../tp_allreduce_l0/embed_spv.cmake
    DEPENDS ${KERNEL_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/../tp_allreduce_l0/embed_spv.cmake
    COMMENT "Embedding SPIR-V -> ${KERNEL_HDR}"
    VERBATIM)

add_executable(${TARGET_NAME}
    tp_l0_call_cost.cpp
    ${KERNEL_HDR})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE LevelZero::LevelZero)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")
