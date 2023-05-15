# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.18)
project(mlas C CXX ASM_MASM)
set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_compile_definitions(BUILD_MLAS_NO_ONNXRUNTIME)
set(CMAKE_CXX_STANDARD 17)
set(ONNXRUNTIME_MLAS_MULTI_ARCH FALSE)
add_library(mlas STATIC
  ${MLAS_SRC_DIR}/platform.cpp
  ${MLAS_SRC_DIR}/threading.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/qgemm.cpp
  ${MLAS_SRC_DIR}/qdwconv.cpp
  ${MLAS_SRC_DIR}/convolve.cpp
  ${MLAS_SRC_DIR}/convsym.cpp
  ${MLAS_SRC_DIR}/pooling.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/reorder.cpp
  ${MLAS_SRC_DIR}/snchwc.cpp
  ${MLAS_SRC_DIR}/activate.cpp
  ${MLAS_SRC_DIR}/logistic.cpp
  ${MLAS_SRC_DIR}/tanh.cpp
  ${MLAS_SRC_DIR}/erf.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/quantize.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
  ${MLAS_SRC_DIR}/qladd.cpp
  ${MLAS_SRC_DIR}/qlmul.cpp
  ${MLAS_SRC_DIR}/qpostprocessor.cpp
  ${MLAS_SRC_DIR}/qlgavgpool.cpp
  ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
)

set(MLAS_LIBS mlas)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${MLAS_SRC_DIR})

function(setup_mlas_source_for_windows)
  #The onnxruntime_target_platform variable was added by Windows AI team in onnxruntime_common.cmake
  #Don't use it for other platforms.
  if(onnxruntime_target_platform STREQUAL "x64")
    set_target_properties(mlas PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
   file(GLOB_RECURSE mlas_platform_srcs_avx CONFIGURE_DEPENDS
     "${MLAS_SRC_DIR}/intrinsics/avx/*.cpp"
   )
   set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

   file(GLOB_RECURSE mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
     "${MLAS_SRC_DIR}/intrinsics/avx2/*.cpp"
   )
   set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
   target_sources(mlas PRIVATE
      ${MLAS_SRC_DIR}/dgemm.cpp
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
      ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8U8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Vnni.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvxVnni.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelM1Avx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/sgemma.asm
      ${MLAS_SRC_DIR}/amd64/cvtfp16a.asm
      ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/LogisticKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TanhKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/ErfKernelFma3.asm
    )
  else()
    target_sources(mlas PRIVATE
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${MLAS_SRC_DIR}/i386/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/i386/SgemmKernelAvx.asm
    )
  endif()
endfunction()

if(MSVC)
    set(onnxruntime_target_platform "x64")
    setup_mlas_source_for_windows()
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
        set(X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(X86_64 TRUE)
    endif()

    #If ONNXRUNTIME_MLAS_MULTI_ARCH is true, we need to go through every if branch below
    #and split MLAS to multiple static libraries.
    #Otherwise, it works like if(...) elseif(...) elseif(...) endif()
    set(MLAS_SOURCE_IS_NOT_SET 1)

    if(X86 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        set(mlas_platform_srcs_sse2
            ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
            ${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
            ${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs
            ${mlas_platform_srcs_sse2}
            ${mlas_platform_srcs_avx}S
        )
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
            set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86_64 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        # Forward the flags for the minimum target platform version from the C
        # compiler to the assembler. This works around CMakeASMCompiler.cmake.in
        # not including the logic to set this flag for the assembler.
        set(CMAKE_ASM${ASM_DIALECT}_OSX_DEPLOYMENT_TARGET_FLAG "${CMAKE_C_OSX_DEPLOYMENT_TARGET_FLAG}")

        # The LLVM assembler does not support the .arch directive to enable instruction
        # set extensions and also doesn't support AVX-512F instructions without
        # turning on support via command-line option. Group the sources by the
        # instruction set extension and explicitly set the compiler flag as appropriate.

        set(mlas_platform_srcs_sse2
            ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
            ${MLAS_SRC_DIR}/x86_64/DgemmKernelSse2.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
            ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
            ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
            ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
            ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
            ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
            ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx.S
            ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx.S
            ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx.S
            ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs_avx2
            ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAvx2.S
            ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx2.S
            ${MLAS_SRC_DIR}/x86_64/QgemmU8U8KernelAvx2.S
            ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvxVnni.S
            ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx2.S
            ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S
            ${MLAS_SRC_DIR}/x86_64/DgemmKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/SconvKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/TransKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/LogisticKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/TanhKernelFma3.S
            ${MLAS_SRC_DIR}/x86_64/ErfKernelFma3.S
            ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
            ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

        set(mlas_platform_srcs_avx512f
            ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx512F.S
            ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
            ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx512F.S
            ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx512F.S
            ${MLAS_SRC_DIR}/x86_64/TransKernelAvx512F.S
            ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

        set(mlas_platform_srcs_avx512core
            ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Core.S
            ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Vnni.S
            ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx512Core.S
            ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx512Core.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mavx512bw -mavx512dq -mavx512vl")

        set(mlas_platform_srcs
            ${MLAS_SRC_DIR}/dgemm.cpp
            ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
            ${mlas_platform_srcs_sse2}
            ${mlas_platform_srcs_avx}
            ${mlas_platform_srcs_avx2}
            ${mlas_platform_srcs_avx512f}
            ${mlas_platform_srcs_avx512core}
        )
        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
            onnxruntime_add_static_library(onnxruntime_mlas_x86_64 ${mlas_platform_srcs})
            set_target_properties(onnxruntime_mlas_x86_64 PROPERTIES OSX_ARCHITECTURES "x86_64")
            list(APPEND MLAS_LIBS onnxruntime_mlas_x86_64)
            set(mlas_platform_srcs )
        else()
            set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH AND MLAS_SOURCE_IS_NOT_SET)
        file(GLOB_RECURSE mlas_platform_srcs
            "${MLAS_SRC_DIR}/scalar/*.cpp")
    endif()
    target_sources(mlas PRIVATE ${mlas_platform_srcs})
endif()

foreach(mlas_target ${MLAS_LIBS})
    target_include_directories(${mlas_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${MLAS_SRC_DIR})
endforeach()

set_target_properties(mlas PROPERTIES FOLDER "MLAS")
if (WIN32)
    target_compile_options(mlas PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/wd6385>" "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>")
endif()
set_property(TARGET mlas PROPERTY POSITION_INDEPENDENT_CODE ON)
