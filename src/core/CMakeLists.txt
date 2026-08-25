# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# The standalone Vulkan core (ov::core::vulkan::cross_platform).
#
# The core is the single runtime for all devices (GPU/CPU/NPU) and platforms
# (desktop Vulkan, MoltenVK, Vulkan SC). Step 5 of VULKAN_CROSS_PLATFORM.md is
# complete: the intel_gpu / cldnn runtime interfaces were removed and the
# backend defines its own minimal runtime types (vk_types.hpp) in the core's
# namespace. There is no dependency on openvino::runtime::dev, openvino::itt,
# oneDNN or any other legacy target.

# The core is written for C++23 (spans, ranges, consteval, ...): pin the
# standard here so the requirement survives global CMake changes. MSVC targets
# use -std:c++latest; non-MSVC compilers get cxx_std_23.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Nothing defines GPU_RT_TYPE anymore (the legacy Vulkan-only runtime branch
# inside intel_gpu selected it); default to VULKAN so the Vulkan engine path
# stays active.
if(NOT DEFINED GPU_RT_TYPE)
    set(GPU_RT_TYPE "VULKAN")
endif()

set(TARGET_NAME "openvino_vulkan_core")

file(GLOB LIBRARY_SOURCES_ALL
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp"
  )

add_library(${TARGET_NAME} STATIC ${LIBRARY_SOURCES_ALL})

# Native SPIR-V kernels: every kernels/*.comp is compiled with
# glslangValidator at build time and embedded into spirv_kernels.inc
# (consumed by vk_kernel_builder.cpp via the builtin native kernel store).
find_program(GLSLANG_VALIDATOR glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator is required to build the native Vulkan kernels. "
                        "Install the Vulkan SDK or add its bin to PATH and re-configure.")
endif()

set(VK_KERNELS_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/spirv")
file(GLOB VK_KERNELS_COMP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.comp")
add_custom_command(
    OUTPUT "${VK_KERNELS_GEN_DIR}/spirv_kernels.inc"
    COMMAND "${CMAKE_COMMAND}"
            -DGLSLANG="${GLSLANG_VALIDATOR}"
            -DSRC="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
            -DOUT="${VK_KERNELS_GEN_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_spirv.cmake"
    DEPENDS ${VK_KERNELS_COMP}
    COMMENT "Compiling native Vulkan kernels to SPIR-V"
)
target_sources(${TARGET_NAME} PRIVATE "${VK_KERNELS_GEN_DIR}/spirv_kernels.inc")
target_include_directories(${TARGET_NAME} PRIVATE "${VK_KERNELS_GEN_DIR}")

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

# The core is self-contained: the source root (for vk_* internal includes) and
# the runtime/ ov-compat layer are the only include roots.
target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CONFIG:Release>:
    $<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os
    >
  >
  $<$<CXX_COMPILER_ID:GNU>:
    $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,6.0>:
      -Wno-ignored-attributes
    >
  >)

if(GPU_RT_TYPE STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED)

    # Optional: embed clspv (OpenCL C -> SPIR-V compiler) into the plugin so that
    # KernelFormat::SOURCE kernels can be compiled at runtime by the Vulkan backend.
    # OV_GPU_VULKAN_CLSPV_PATH must point to a clspv build tree, OV_GPU_VULKAN_LLVM_PATH
    # to the LLVM build tree clspv was built against.
    set(OV_GPU_VULKAN_CLSPV_PATH "" CACHE PATH "Path to a clspv build tree (optional, enables OpenCL C -> SPIR-V compilation in the Vulkan runtime)")
    set(OV_GPU_VULKAN_LLVM_PATH "" CACHE PATH "Path to the LLVM build tree used to build clspv (optional)")
    if(OV_GPU_VULKAN_CLSPV_PATH)
        target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_CLSPV)
        set(CLSPV_STATIC_LIBS
            "${OV_GPU_VULKAN_CLSPV_PATH}/lib/libclspv_core.a"
            "${OV_GPU_VULKAN_CLSPV_PATH}/third_party/SPIRV-Tools/source/libSPIRV-Tools.a")
        if(OV_GPU_VULKAN_LLVM_PATH)
            file(GLOB LLVM_STATIC_LIBS "${OV_GPU_VULKAN_LLVM_PATH}/lib/lib*.a")
            list(APPEND CLSPV_STATIC_LIBS ${LLVM_STATIC_LIBS})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            # LLVM static libs have cyclic dependencies, resolve them with --start-group
            target_link_options(${TARGET_NAME} PRIVATE "-Wl,--start-group")
            foreach(clspv_lib IN LISTS CLSPV_STATIC_LIBS)
                target_link_libraries(${TARGET_NAME} PRIVATE ${clspv_lib})
            endforeach()
            target_link_options(${TARGET_NAME} PRIVATE "-Wl,--end-group")
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE ${CLSPV_STATIC_LIBS})
        endif()
        if(WIN32)
            # System libraries required by LLVM Support on Windows
            target_link_libraries(${TARGET_NAME} PRIVATE ole32 uuid ntdll advapi32 shell32 psapi imagehlp version ws2_32 userenv bcrypt)
        endif()
    endif()

    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if((NOT ANDROID) AND (UNIX))
    target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

