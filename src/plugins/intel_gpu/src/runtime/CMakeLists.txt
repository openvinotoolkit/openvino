# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# OV_GPU_SUFFIX (empty for single-RT builds) distinguishes the per-runtime target names of a
# COMBINED build; OV_GPU_RT is this pass's runtime (defaults to the global GPU_RT_TYPE).
set(TARGET_NAME "openvino_intel_gpu_runtime${OV_GPU_SUFFIX}")
if(NOT DEFINED OV_GPU_RT)
    # Reachable only if this directory is added outside ov_gpu_build_stack(), which always sets a
    # single runtime; COMBINED names two, so a stack built from it would belong to neither.
    if(GPU_RT_TYPE STREQUAL "COMBINED")
        message(FATAL_ERROR "OV_GPU_RT must be set to a single runtime before adding this directory; got COMBINED")
    endif()
    set(OV_GPU_RT "${GPU_RT_TYPE}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)

set(LIBRARY_SOURCES_MAIN ${GPU_RUNTIME_MAIN_SOURCES})
set(LIBRARY_SOURCES_OCL ${GPU_RUNTIME_OCL_SOURCES})
set(LIBRARY_SOURCES_ZE ${GPU_RUNTIME_ZE_SOURCES} ${GPU_RUNTIME_ZE_COMPUTE_RUNTIME_HEADERS})
set(LIBRARY_SOURCES_SYCL ${GPU_RUNTIME_SYCL_SOURCES})
set(OCL_SYCL_SOURCES ${GPU_RUNTIME_OCL_SYCL_SOURCES})

# ocl/sycl_*.{h,hpp,cpp} require the SYCL compiler, so they are only part of the
# OCL runtime's source set when SYCL support is enabled.
if(OV_GPU_WITH_SYCL)
    list(APPEND LIBRARY_SOURCES_OCL ${OCL_SYCL_SOURCES})
endif()

set(LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_MAIN})

if(OV_GPU_RT STREQUAL "OCL")
    list(APPEND LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_OCL})
    set(SYCL_SOURCES ${OCL_SYCL_SOURCES})
endif()

if(OV_GPU_RT STREQUAL "ZE")
    list(APPEND LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_OCL})
    list(APPEND LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_ZE})
endif()

if(OV_GPU_RT STREQUAL "SYCL")
    list(APPEND LIBRARY_SOURCES_ALL ${LIBRARY_SOURCES_SYCL})
    set(SYCL_SOURCES ${LIBRARY_SOURCES_SYCL})
endif()

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

if(OV_GPU_WITH_SYCL)
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SYCL_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_SYCL)
endif()

if(GPU_RT_TYPE STREQUAL "SYCL")
    find_path(OCLOC_API_INCLUDE_DIR NAMES ocloc_api.h REQUIRED)
    target_include_directories(${TARGET_NAME} PRIVATE ${OCLOC_API_INCLUDE_DIR})
endif()

if(ENABLE_ONEDNN_FOR_GPU)
  ov_target_link_libraries_as_system(${TARGET_NAME} PUBLIC onednn_gpu_tgt${OV_GPU_SUFFIX})
endif()

ov_set_threading_interface_for(${TARGET_NAME})
ov_gpu_set_runtime_interface_for(${TARGET_NAME} RUNTIME ${OV_GPU_RT})

# Requires OpenCL even for Level Zero builds because of kernel builder and OneDNN
target_link_libraries(${TARGET_NAME} PRIVATE
    OpenCL::OpenCL
    openvino::itt
    openvino::runtime::dev
  )

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
