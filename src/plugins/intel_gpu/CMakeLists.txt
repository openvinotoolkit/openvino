# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# convert white-space separated string to list
string(REPLACE " " ";" SYCL_EXTRA_FLAGS "${SYCL_EXTRA_FLAGS}")

if(OV_COMPILER_IS_INTEL_LLVM)
    # For windows we need to disable warning as error option to make FindSYCL.cmake work
    if (WIN32)
        ov_add_compiler_flags(/WX-)
    endif()

    find_package(IntelSYCL REQUIRED)
    set(OV_GPU_WITH_SYCL 1)
    set_property(TARGET IntelSYCL::SYCL_CXX APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS ${SYCL_EXTRA_FLAGS}
    )
    set_property(TARGET IntelSYCL::SYCL_CXX APPEND PROPERTY
        INTERFACE_LINK_OPTIONS ${SYCL_EXTRA_FLAGS}
    )

    if (WIN32)
        ov_add_compiler_flags(/WX)
        ov_add_compiler_flags(/Wno-ignored-attributes)
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(OV_COMPILER_IS_CLANG)
    include(CMakePushCheckState)
    include(CheckCXXSourceCompiles)

    set(FSYCL_OPTION -fsycl)
    check_cxx_compiler_flag(${FSYCL_OPTION} HAS_FSYCL_OPTION)

    if(HAS_FSYCL_OPTION)
        set(CHECK_SYCL_HEADER_SOURCE
        "
            #include <sycl/sycl.hpp>
            int main() {
                return 0;
            }
        ")

        cmake_push_check_state(RESET)
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${FSYCL_OPTION} ${SYCL_EXTRA_FLAGS})
        check_cxx_source_compiles("${CHECK_SYCL_HEADER_SOURCE}" HAS_SYCL_HEADER_WITH_FSYCL)
        cmake_pop_check_state()
    endif()

    if(HAS_FSYCL_OPTION AND HAS_SYCL_HEADER_WITH_FSYCL)
        set(OV_GPU_WITH_SYCL 1)

        function(add_sycl_to_target)
            set(options)
            set(one_value_args TARGET)
            set(multi_value_args SOURCES)
            cmake_parse_arguments(SYCL
                ""
                "${one_value_args}"
                "${multi_value_args}"
                ${ARGN})
            if(SYCL_SOURCES)
                foreach(source ${SYCL_SOURCES})
                    set_property(SOURCE ${source}
                                 APPEND
                                 PROPERTY COMPILE_OPTIONS
                                 ${FSYCL_OPTION} ${SYCL_EXTRA_FLAGS}
                    )
                endforeach()
            else()
                target_compile_options(${SYCL_TARGET} PUBLIC ${FSYCL_OPTION} ${SYCL_EXTRA_FLAGS})
            endif()
            target_link_options(${SYCL_TARGET} PUBLIC ${FSYCL_OPTION} ${SYCL_EXTRA_FLAGS})
        endfunction()
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
    # see https://github.com/oneapi-src/oneDNN/issues/2028
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

if(GPU_RT_TYPE STREQUAL "SYCL" AND NOT OV_GPU_WITH_SYCL)
    message(FATAL_ERROR "GPU_RT_TYPE is set to SYCL, but SYCL is not available for the selected compiler/environment.")
endif()

# When -fsycl and -flto are both active, archive members are emitted in a format
# that the linker cannot read from static libraries ("archive member is neither ET_REL nor LLVM bitcode").
if(OV_GPU_WITH_SYCL AND ENABLE_LTO)
    message(FATAL_ERROR "Intel GPU plugin with SYCL is not supported with ENABLE_LTO=ON")
endif()

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(ENABLE_GPU_DEBUG_CAPS)
    add_definitions(-DGPU_DEBUG_CONFIG=1)
    add_definitions(-DENABLE_DEBUG_CAPS=1)
endif()

set(INTEL_GPU_TARGET_OCL_VERSION "300"
  CACHE STRING "Target version of OpenCL which should be used by GPU plugin")

add_definitions(-DCL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(thirdparty)
include(thirdparty/cmake/rapidjson.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Werror)
endif()

file(GLOB_RECURSE PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_gpu/plugin/*.hpp)

# Build one full GPU stack (runtime + kernel_selector + graph + plugin) for runtime RT_TYPE.
# SUFFIX distinguishes a COMBINED build's two stacks (empty for the ZE/default twin, "_ocl"
# for the OCL twin) and single-RT builds (empty). MODULE_NAME is the plugin library name.
# EXTRA_PLUGIN_ARGS are forwarded to ov_add_plugin (e.g. DISPATCH_GROUP for COMBINED).
function(ov_gpu_build_stack RT_TYPE SUFFIX MODULE_NAME)
    set(OV_GPU_RT "${RT_TYPE}")
    set(OV_GPU_SUFFIX "${SUFFIX}")

    # Distinct binary dirs so the same source dirs can be added twice in a COMBINED build.
    add_subdirectory(src/runtime         runtime${SUFFIX})
    add_subdirectory(src/kernel_selector kernel_selector${SUFFIX})
    add_subdirectory(src/graph           graph${SUFFIX})

    ov_add_plugin(NAME ${MODULE_NAME}
                  DEVICE_NAME "GPU"
                  SOURCES ${PLUGIN_SOURCES}
                  DEFAULT_CONFIG ${PLUGIN_DEFAULT_CONFIG}
                  VERSION_DEFINES_FOR src/plugin/plugin.cpp
                  # ADD_CLANG_TIDY makes ov_add_plugin apply clang-tidy to this target.
                  # Libraries not created via ov_add_plugin call ov_apply_clang_tidy() directly.
                  ADD_CLANG_TIDY
                  ${ARGN})

    target_compile_options(${MODULE_NAME} PRIVATE
        $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

    target_link_libraries(${MODULE_NAME} PRIVATE openvino_intel_gpu_graph${SUFFIX} openvino::pugixml)

    target_include_directories(${MODULE_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include/)

    ov_set_threading_interface_for(${MODULE_NAME})
    ov_gpu_set_runtime_interface_for(${MODULE_NAME} RUNTIME ${RT_TYPE})

    set_target_properties(${MODULE_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

    # Workaround to avoid warnings during LTO build
    if(CMAKE_COMPILER_IS_GNUCXX)
      set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow"
                                                      LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow")
    endif()

    ov_build_target_faster(${MODULE_NAME} PCH)
endfunction()

if(GPU_RT_TYPE STREQUAL "COMBINED")
    # Two libraries under one device name (a dispatch group). OCL is registered first so the merged
    # device ids follow its enumeration, as today; ZE keeps the default-named targets tests link.
    ov_gpu_build_stack(OCL "_ocl" ${TARGET_NAME}_ocl  DISPATCH_GROUP)
    ov_gpu_build_stack(ZE  ""     ${TARGET_NAME}      DISPATCH_GROUP)
else()
    ov_gpu_build_stack(${GPU_RT_TYPE} "" ${TARGET_NAME})
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
