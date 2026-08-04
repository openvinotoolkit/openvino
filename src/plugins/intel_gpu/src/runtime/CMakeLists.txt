# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_runtime")

file(GLOB LIBRARY_SOURCES_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

file(GLOB LIBRARY_SOURCES_VULKAN
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.cpp"
)

set(LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_MAIN} ${LIBRARY_SOURCES_VULKAN})

add_library(${TARGET_NAME} STATIC ${LIBRARY_SOURCES_ALL})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${INCLUDE_DIR}>)

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
    target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_VULKAN_RT)

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
endif()

if(ENABLE_ONEDNN_FOR_GPU)
  ov_target_link_libraries_as_system(${TARGET_NAME} PUBLIC onednn_gpu_tgt)
endif()

ov_set_threading_interface_for(${TARGET_NAME})
ov_gpu_set_runtime_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::itt
    openvino::runtime::dev
  )

if(GPU_RT_TYPE STREQUAL "VULKAN")
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

# Static library built via ov_add_target, so apply clang-tidy directly
# (the ADD_CLANG_TIDY option only exists for plugins created via ov_add_plugin).
ov_apply_clang_tidy(TARGET ${TARGET_NAME})

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
