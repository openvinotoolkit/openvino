# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Minimal Level Zero sample: tp_allreduce between two discrete GPUs
# using a single shared L0 context and no host-side data transfer.

set(TARGET_NAME tp_allreduce_l0)

# ---------------------------------------------------------------------------
# Locate Level Zero.  `LevelZero::LevelZero` is provided by the in-tree
# thirdparty build, so the sample follows the same loader the plugin uses.
# ---------------------------------------------------------------------------
if(NOT TARGET LevelZero::LevelZero)
    message(STATUS "tp_allreduce_l0: Level Zero not available; sample disabled")
    return()
endif()

# ---------------------------------------------------------------------------
# Compile the OpenCL kernel to SPIR-V at configure/build time using ocloc.
# ---------------------------------------------------------------------------
find_program(OCLOC_EXE ocloc)
if(NOT OCLOC_EXE)
    message(STATUS "tp_allreduce_l0: 'ocloc' (Intel offline compiler) not found; sample disabled")
    return()
endif()

# The sample reduces f16 only, but compiles the plugin's kernel source rather
# than a copy of it, so the two cannot drift apart.
set(KERNEL_SRC  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/allreduce_sum.cl)
set(KERNEL_SPV  ${CMAKE_CURRENT_BINARY_DIR}/allreduce_sum.spv)
set(KERNEL_HDR  ${CMAKE_CURRENT_BINARY_DIR}/allreduce_sum_spv.h)

# Generate SPIR-V only (no target binary).  '-device' is required by ocloc
# even with '-spv_only'; the chosen device only influences any builtin
# linking, not the SPIR-V output, so any modern target works.
# '-output_no_suffix' prevents ocloc from appending '_<device>' to the file.
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

# Embed the SPIR-V blob as a C array.
add_custom_command(
    OUTPUT  ${KERNEL_HDR}
    COMMAND ${CMAKE_COMMAND}
            -DSPV_FILE=${KERNEL_SPV}
            -DHDR_FILE=${KERNEL_HDR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_spv.cmake
    DEPENDS ${KERNEL_SPV} ${CMAKE_CURRENT_SOURCE_DIR}/embed_spv.cmake
    COMMENT "Embedding SPIR-V -> ${KERNEL_HDR}"
    VERBATIM)

# ---------------------------------------------------------------------------
# Sample executable
# ---------------------------------------------------------------------------
add_executable(${TARGET_NAME}
    tp_allreduce_l0.cpp
    ${KERNEL_HDR})

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE LevelZero::LevelZero)

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OpenVINO_SOURCE_DIR}/bin/intel64/$<CONFIG>")
