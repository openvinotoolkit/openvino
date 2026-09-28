# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_ocl_v2_obj${OV_GPU_SUFFIX}")

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

add_custom_target(run_ocl_codegen${OV_GPU_SUFFIX} ALL DEPENDS "${CODEGEN_INCDIR}/${KERNEL_SOURCES}")

# TernOCL int2 kernels (ternocl_int2/), embedded as source strings. They come from
# the thirdparty/TernOCL submodule (github.com/libxsmm/TernOCL); TERNOCL_ROOT
# (cache or environment) points at another checkout, e.g. while editing kernels.
if(NOT DEFINED TERNOCL_ROOT)
    if(DEFINED ENV{TERNOCL_ROOT})
        set(TERNOCL_ROOT "$ENV{TERNOCL_ROOT}")
    else()
        set(TERNOCL_ROOT "${MAIN_DIR}/thirdparty/TernOCL")
    endif()
endif()
set(TERNOCL_ROOT "${TERNOCL_ROOT}" CACHE PATH "TernOCL kernel collection used by the int2 FullyConnected impl")
set(TERNOCL_FILES "${TERNOCL_ROOT}/int2_fp16_upcvt/int2_fp16_upcvt.cl"
                  "${TERNOCL_ROOT}/int2_via_int2_x_int8_dpas/int2_int8_dpas.cl"
                  "${TERNOCL_ROOT}/common/epilogue.clh"
                  "${TERNOCL_ROOT}/hadamard/hadamard_fwht.cl")
foreach(f IN LISTS TERNOCL_FILES)
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "TernOCL kernel ${f} not found: run `git submodule update --init "
                            "src/plugins/intel_gpu/thirdparty/TernOCL` or set TERNOCL_ROOT")
    endif()
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${TERNOCL_FILES})
# Emits a source as concatenated raw-string chunks: MSVC caps a single literal at 16 KB.
function(ternocl_literal out_var name text)
    string(LENGTH "${text}" len)
    set(lit "static const char ${name}[] =")
    set(pos 0)
    while(pos LESS len)
        string(SUBSTRING "${text}" ${pos} 8000 chunk)
        string(APPEND lit "\n    R\"TERNOCL(${chunk})TERNOCL\"")
        math(EXPR pos "${pos} + 8000")
    endwhile()
    set(${out_var} "${lit};\n" PARENT_SCOPE)
endfunction()
file(READ "${TERNOCL_ROOT}/common/epilogue.clh" TERNOCL_EPILOGUE)
file(READ "${TERNOCL_ROOT}/int2_fp16_upcvt/int2_fp16_upcvt.cl" TERNOCL_UPCVT)
string(REPLACE "#include \"epilogue.clh\"" "${TERNOCL_EPILOGUE}" TERNOCL_UPCVT "${TERNOCL_UPCVT}")
file(READ "${TERNOCL_ROOT}/int2_via_int2_x_int8_dpas/int2_int8_dpas.cl" TERNOCL_INT8)
string(REPLACE "#include \"epilogue.clh\"" "${TERNOCL_EPILOGUE}" TERNOCL_INT8 "${TERNOCL_INT8}")
file(READ "${TERNOCL_ROOT}/hadamard/hadamard_fwht.cl" TERNOCL_FWHT)
ternocl_literal(TERNOCL_UPCVT_LIT kTernoclUpcvtSource "${TERNOCL_UPCVT}")
ternocl_literal(TERNOCL_INT8_LIT kTernoclInt8Source "${TERNOCL_INT8}")
ternocl_literal(TERNOCL_FWHT_LIT kTernoclFwhtSource "${TERNOCL_FWHT}")
file(WRITE "${CODEGEN_CACHE_DIR}/ternocl_int2_kernels.inc"
     "// Generated from ${TERNOCL_ROOT} by impls/ocl_v2/CMakeLists.txt; do not edit.\n#pragma once\n"
     "${TERNOCL_UPCVT_LIT}${TERNOCL_INT8_LIT}${TERNOCL_FWHT_LIT}")
configure_file("${CODEGEN_CACHE_DIR}/ternocl_int2_kernels.inc" "${CODEGEN_INCDIR}/ternocl_int2_kernels.inc" COPYONLY)

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    INCLUDES $<BUILD_INTERFACE:${CODEGEN_INCDIR}>
    BYPASS
        ADDITIONAL_SOURCE_DIRS ${CODEGEN_INCDIR}
        DEPENDENCIES run_ocl_codegen${OV_GPU_SUFFIX}
        ADD_CLANG_FORMAT
)

ov_build_target_faster(${TARGET_NAME} PCH)

if(OV_GPU_RT STREQUAL "SYCL")
    # Some sources include oneDNN headers that require SYCL backend support.
    # Without the SYCL compilation flag, the compiler will fail with "Unsupported compiler" error.
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES)
endif()
