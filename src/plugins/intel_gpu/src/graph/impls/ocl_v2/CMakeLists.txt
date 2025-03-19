# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_ocl_v2_obj")

# Path which points to root directory where code generated elements are created
# (specific to build configuration).
set(CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/codegen")
set(CODEGEN_CACHE_DIR  "${CODEGEN_DIR}/cache")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(CODEGEN_INCDIR  "${CODEGEN_DIR}/include")

set(KERNEL_SOURCES "gpu_ocl_kernel_sources.inc")
set(KERNEL_HEADERS "gpu_ocl_kernel_headers.inc")

set(CODEGEN_CACHE_SOURCES "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
                          "${CODEGEN_INCDIR}/${KERNEL_HEADERS}")

file(GLOB_RECURSE KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/*.cl")

set_property(SOURCE ${CODEGEN_CACHE_SOURCES} PROPERTY GENERATED TRUE)

add_custom_command(OUTPUT "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CODEGEN_CACHE_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${CODEGEN_SCRIPT}" -out_sources "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}"
                                                      -out_headers "${CODEGEN_CACHE_DIR}/${KERNEL_HEADERS}"
                                                      -in_kernels_dir "${CMAKE_CURRENT_SOURCE_DIR}/"
                                                      -in_headers_dir "${MAIN_DIR}/src/kernel_selector/cl_kernels/include"
                                                      -lang "ocl"
  DEPENDS ${KERNELS} "${CODEGEN_SCRIPT}"
  COMMENT "Generating ${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES} ..."
)

add_custom_command(OUTPUT "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}" "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${KERNEL_HEADERS}" "${CODEGEN_INCDIR}/${KERNEL_HEADERS}"
  DEPENDS "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}" "${KERNELS}" "${CODEGEN_SCRIPT}"
  COMMENT "Updating file if the file changed (${CODEGEN_INCDIR}/${KERNEL_SOURCES}) ..."
)

add_custom_target(run_ocl_codegen ALL DEPENDS "${CODEGEN_INCDIR}/${KERNEL_SOURCES}")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    INCLUDES $<BUILD_INTERFACE:${CODEGEN_INCDIR}>
    BYPASS
        ADDITIONAL_SOURCE_DIRS ${CODEGEN_INCDIR}
        DEPENDENCIES run_ocl_codegen
        ADD_CLANG_FORMAT
)
