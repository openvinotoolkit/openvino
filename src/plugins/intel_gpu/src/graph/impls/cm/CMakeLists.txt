# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_cm_obj")

# Path which points to root directory where code generated elements are created
# (specific to build configuration).
set(CODEGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/codegen")
set(CODEGEN_CACHE_DIR "${CODEGEN_DIR}/cache")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(CODEGEN_INCDIR  "${CODEGEN_DIR}/include")

set(KERNEL_SOURCES "gpu_cm_kernel_sources.inc")
set(KERNEL_HEADERS "gpu_cm_kernel_headers.inc")

set(CODEGEN_CACHE_SOURCES "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
                          "${CODEGEN_INCDIR}/${KERNEL_HEADERS}")

file(GLOB_RECURSE KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/*.cm")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

set_property(SOURCE ${CODEGEN_CACHE_SOURCES} PROPERTY GENERATED TRUE)

set(XETLA_HEADER "cm_xetla.h")
set(XETLA_HEADER_FULL_PATH "${CODEGEN_CACHE_DIR}/cm_kernels/include/batch_headers/${XETLA_HEADER}")
if(WIN32)
    set(PREPROCESSOR_OPTIONS -D _WIN32 -EP)
else()
    set(PREPROCESSOR_OPTIONS  -E -P)
endif()

add_custom_command(OUTPUT "${CODEGEN_CACHE_DIR}/cm_kernels"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CODEGEN_CACHE_DIR}/cm_kernels/"
)

foreach(KERNEL IN LISTS KERNELS)
    get_filename_component(FILENAME ${KERNEL} NAME)
    add_custom_command(
        OUTPUT "${CODEGEN_CACHE_DIR}/cm_kernels/${FILENAME}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${KERNEL} "${CODEGEN_CACHE_DIR}/cm_kernels/${FILENAME}"
        DEPENDS ${KERNEL} "${CODEGEN_CACHE_DIR}/cm_kernels"
        COMMENT "Copying ${FILE} ${CODEGEN_CACHE_DIR}/cm_kernels"
    )
    list(APPEND COPIED_KERNELS "${CODEGEN_CACHE_DIR}/cm_kernels/${FILENAME}")
endforeach()

add_custom_command(OUTPUT "${XETLA_HEADER_FULL_PATH}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CODEGEN_CACHE_DIR}/cm_kernels/include/"
    COMMAND "${CMAKE_CXX_COMPILER}" "${XETLA_HEADER_FULL_PATH}" ${PREPROCESSOR_OPTIONS} -I ${XETLA_INCLUDE_DIR} > "${CODEGEN_CACHE_DIR}/${XETLA_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E rename "${CODEGEN_CACHE_DIR}/${XETLA_HEADER}" "${XETLA_HEADER_FULL_PATH}"
    DEPENDS "${COPIED_KERNELS}" "${HEADERS}"
    COMMENT "Copying CM headers and preprocessing XeTLA headers ..."
)

add_custom_command(OUTPUT "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CODEGEN_CACHE_DIR}"
  COMMAND "${Python3_EXECUTABLE}" "${CODEGEN_SCRIPT}" -out_sources "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}"
                                                      -out_headers "${CODEGEN_CACHE_DIR}/${KERNEL_HEADERS}"
                                                      -in_kernels_dir "${CODEGEN_CACHE_DIR}/cm_kernels"
                                                      -in_headers_dir "${CODEGEN_CACHE_DIR}/cm_kernels/include"
                                                      -lang "cm"
  DEPENDS ${KERNELS} "${CODEGEN_SCRIPT}" "${XETLA_HEADER_FULL_PATH}"
  COMMENT "Generating ${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES} ..."
)

add_custom_command(OUTPUT "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}" "${CODEGEN_INCDIR}/${KERNEL_SOURCES}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHE_DIR}/${KERNEL_HEADERS}" "${CODEGEN_INCDIR}/${KERNEL_HEADERS}"
  DEPENDS "${CODEGEN_CACHE_DIR}/${KERNEL_SOURCES}" "${KERNELS}" "${HEADERS}" "${CODEGEN_SCRIPT}"
  COMMENT "Updating file if the file changed (${CODEGEN_INCDIR}/${KERNEL_SOURCES}) ..."
)

add_custom_target(run_cm_codegen ALL DEPENDS "${CODEGEN_INCDIR}/${KERNEL_SOURCES}")

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    INCLUDES $<BUILD_INTERFACE:${CODEGEN_INCDIR}>
    BYPASS
        ADDITIONAL_SOURCE_DIRS ${CODEGEN_INCDIR}
        EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DEPENDENCIES run_cm_codegen
        ADD_CLANG_FORMAT
)
