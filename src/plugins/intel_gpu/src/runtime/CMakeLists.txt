# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_runtime")

file(GLOB LIBRARY_SOURCES_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

file(GLOB LIBRARY_SOURCES_OCL
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.cpp"
)

file(GLOB LIBRARY_SOURCES_ZE
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/compute_runtime/*.h"
)

file(GLOB LIBRARY_SOURCES_SYCL
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/*.cpp"
)

file(GLOB LIBRARY_SOURCES_SYCL_INTEL
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/intel/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/intel/*.cpp"
)
list(APPEND LIBRARY_SOURCES_SYCL ${LIBRARY_SOURCES_SYCL_INTEL})

file(GLOB LIBRARY_SOURCES_VULKAN
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.cpp"
)

file(GLOB_RECURSE OCL_SYCL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ocl/sycl_*.[ch]pp")

if(NOT OV_GPU_WITH_SYCL)
    list(REMOVE_ITEM LIBRARY_SOURCES_OCL ${OCL_SYCL_SOURCES})
endif()

if(OV_GPU_RUNTIME_OCL_ENABLED OR OV_GPU_RUNTIME_ZE_ENABLED)
    list(APPEND SYCL_SOURCES ${OCL_SYCL_SOURCES})
endif()

if(OV_GPU_RUNTIME_SYCL_ENABLED)
    list(APPEND SYCL_SOURCES ${LIBRARY_SOURCES_SYCL})
endif()

list(REMOVE_DUPLICATES SYCL_SOURCES)

function(ov_gpu_configure_runtime_target TARGET)
    target_include_directories(${TARGET} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${INCLUDE_DIR}>)
    # Keep runtime OBJECT libraries out of the static plugin registration cycle.
    # They need OpenVINO headers for compilation, while the final archive owns
    # the actual openvino::runtime::dev link dependency.
    target_include_directories(${TARGET} PRIVATE
        $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

    target_compile_options(${TARGET} PRIVATE
      $<$<CONFIG:Release>:
        $<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os
        >
      >
      $<$<CXX_COMPILER_ID:GNU>:
        $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,6.0>:
          -Wno-ignored-attributes
        >
      >)

    target_link_libraries(${TARGET} PRIVATE openvino::itt)
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${TARGET} PRIVATE OPENVINO_STATIC_LIBRARY)
    endif()
    if(ENABLE_ONEDNN_FOR_GPU)
        ov_target_link_libraries_as_system(${TARGET} PUBLIC onednn_gpu_tgt)
    endif()
    ov_set_threading_interface_for(${TARGET})
    ov_gpu_set_runtime_definitions_for(${TARGET})
    set_target_properties(${TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
    ov_build_target_faster(${TARGET} PCH)
    ov_apply_clang_tidy(TARGET ${TARGET})
endfunction()

set(RUNTIME_BACKEND_OBJECTS "")

if(OV_GPU_RUNTIME_OCL_ENABLED OR OV_GPU_RUNTIME_ZE_ENABLED)
    add_library(openvino_intel_gpu_runtime_ocl_obj OBJECT ${LIBRARY_SOURCES_OCL})
    ov_gpu_configure_runtime_target(openvino_intel_gpu_runtime_ocl_obj)
    ov_gpu_set_opencl_api_version_for(openvino_intel_gpu_runtime_ocl_obj)
    target_link_libraries(openvino_intel_gpu_runtime_ocl_obj PRIVATE OpenCL::OpenCL)
    if(OV_GPU_RUNTIME_ZE_ENABLED)
        target_link_libraries(openvino_intel_gpu_runtime_ocl_obj PRIVATE openvino::zero_loader)
    endif()
    list(APPEND RUNTIME_BACKEND_OBJECTS $<TARGET_OBJECTS:openvino_intel_gpu_runtime_ocl_obj>)
endif()

if(OV_GPU_RUNTIME_ZE_ENABLED)
    add_library(openvino_intel_gpu_runtime_ze_obj OBJECT ${LIBRARY_SOURCES_ZE})
    ov_gpu_configure_runtime_target(openvino_intel_gpu_runtime_ze_obj)
    ov_gpu_set_opencl_api_version_for(openvino_intel_gpu_runtime_ze_obj)
    target_link_libraries(openvino_intel_gpu_runtime_ze_obj PRIVATE
        OpenCL::OpenCL
        openvino::zero_loader)
    list(APPEND RUNTIME_BACKEND_OBJECTS $<TARGET_OBJECTS:openvino_intel_gpu_runtime_ze_obj>)
endif()

if(OV_GPU_RUNTIME_SYCL_ENABLED)
    add_library(openvino_intel_gpu_runtime_sycl_obj OBJECT ${LIBRARY_SOURCES_SYCL})
    ov_gpu_configure_runtime_target(openvino_intel_gpu_runtime_sycl_obj)
    find_path(OCLOC_API_INCLUDE_DIR NAMES ocloc_api.h REQUIRED)
    target_include_directories(openvino_intel_gpu_runtime_sycl_obj PRIVATE ${OCLOC_API_INCLUDE_DIR})
    target_link_libraries(openvino_intel_gpu_runtime_sycl_obj PRIVATE LevelZero::Headers)
    add_sycl_to_target(TARGET openvino_intel_gpu_runtime_sycl_obj SOURCES ${LIBRARY_SOURCES_SYCL})
    list(APPEND RUNTIME_BACKEND_OBJECTS $<TARGET_OBJECTS:openvino_intel_gpu_runtime_sycl_obj>)
endif()

if(OV_GPU_RUNTIME_VULKAN_ENABLED)
    add_library(openvino_intel_gpu_runtime_vulkan_obj OBJECT ${LIBRARY_SOURCES_VULKAN})
    ov_gpu_configure_runtime_target(openvino_intel_gpu_runtime_vulkan_obj)
    target_link_libraries(openvino_intel_gpu_runtime_vulkan_obj PRIVATE Vulkan::Vulkan)
    list(APPEND RUNTIME_BACKEND_OBJECTS $<TARGET_OBJECTS:openvino_intel_gpu_runtime_vulkan_obj>)
endif()

add_library(${TARGET_NAME} STATIC ${LIBRARY_SOURCES_MAIN} ${RUNTIME_BACKEND_OBJECTS})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_gpu_configure_runtime_target(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::dev)

# Keep Vulkan backend-selection macros on the single composition source that
# consumes them. Other runtime and GPU sources receive no Vulkan preprocessor state.
if(OV_GPU_RUNTIME_VULKAN_ENABLED)
    set_property(SOURCE runtime_backend_registry.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS OV_GPU_WITH_VULKAN_RT=1)
endif()
if(GPU_RT_TYPE STREQUAL "VULKAN")
    set_property(SOURCE runtime_backend_registry.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS OV_GPU_DEFAULT_VULKAN_RT=1)
endif()

# Object-library link dependencies do not propagate when their objects are
# collected into this static archive. Keep the backend loader after the archive
# on final link lines so linkers using --as-needed retain the runtime dependency.
if(OV_GPU_RUNTIME_ZE_ENABLED)
    # The ZE runtime owns OpenCL interoperability code and calls the OpenCL
    # loader even when ZE is the selected execution backend. Keep both loaders
    # in the archive link interface so static-link ordering follows the runtime.
    target_link_libraries(${TARGET_NAME} PRIVATE
        OpenCL::OpenCL
        openvino::zero_loader)
endif()
if(OV_GPU_RUNTIME_VULKAN_ENABLED)
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
endif()

if(OV_GPU_WITH_SYCL)
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SYCL_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_SYCL)
endif()

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
