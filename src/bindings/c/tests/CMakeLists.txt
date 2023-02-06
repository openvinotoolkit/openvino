# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# OpenVINO Legacy C API test sample
set(TARGET_NAME "InferenceEngineCAPITests")

add_executable(${TARGET_NAME} ie_c_api_test.cpp test_model_repo.hpp)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_c commonTestUtils gtest_main)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    $<$<BOOL:${ENABLE_GAPI_PREPROCESSING}>:ENABLE_GAPI_PREPROCESSING>
    DATA_PATH=\"${DATA_PATH}\"
    MODELS_PATH=\"${MODELS_PATH}\")

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    add_dependencies(${TARGET_NAME} openvino_auto_batch_plugin)
endif()

if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)

# OpenVINO 2.0 and Legacy C API test sample
set(TARGET_NAME "ov_capi_test")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ov_*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL is disabled or not found, ${TARGET_NAME} will be built without OpenCL support. Install OpenCL.")
endif()

set(opencl_header_search_params
    HINTS
    ${opencl_root_hints}
    PATHS
    ENV "PROGRAMFILES(X86)"
    ENV AMDAPPSDKROOT
    ENV INTELOCLSDKROOT
    ENV NVSDKCOMPUTE_ROOT
    ENV CUDA_PATH
    ENV ATISTREAMSDKROOT
    ENV OCL_ROOT
    PATH_SUFFIXES
    "include"
    "OpenCL/common/inc"
    "AMD APP/include")

find_path(OpenCL_HPP_INCLUDE_DIR
    NAMES
    CL/opencl.hpp OpenCL/opencl.hpp
    ${opencl_header_search_params})

find_path(CL2_HPP_INCLUDE_DIR
    NAMES
    CL/cl2.hpp OpenCL/cl2.hpp
    ${opencl_header_search_params})

if(OpenCL_FOUND AND(OpenCL_HPP_INCLUDE_DIR OR CL2_HPP_INCLUDE_DIR))
    set(OpenCL_DEFINITIONS HAVE_GPU_DEVICE_MEM_SUPPORT)

    if(OpenCL_HPP_INCLUDE_DIR)
        list(APPEND OpenCL_HEADERS ${OpenCL_HPP_INCLUDE_DIR})

        # the macro below is defined when opencl.hpp is found to suppress deprecation message from cl2.hpp
        list(APPEND OpenCL_DEFINITIONS OV_GPU_USE_OPENCL_HPP)
    endif()

    if(CL2_HPP_INCLUDE_DIR)
        list(APPEND OpenCL_HEADERS ${CL2_HPP_INCLUDE_DIR})
    endif()

    set(OpenCL_LIB OpenCL::OpenCL)

    # cmake cannot set properties for imported targets
    get_target_property(opencl_target OpenCL::OpenCL ALIASED_TARGET)

    if(NOT TARGET ${opencl_target})
        set(opencl_target OpenCL::OpenCL)
    endif()

    set_property(TARGET ${opencl_target} APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_HEADERS})
    set_property(TARGET ${opencl_target} APPEND PROPERTY
        INTERFACE_COMPILE_DEFINITIONS ${OpenCL_DEFINITIONS})
else()
    message(WARNING "OpenCL CPP header is not found, ${TARGET_NAME} will be built without OpenCL support. Download it from: https://github.com/KhronosGroup/OpenCL-CLHPP and set -Dopencl_root_hints=[PATH]/OpenCL-CLHPP/include to cmake.")
endif()

if(NOT OpenCL_FOUND OR NOT OpenCL_HEADERS)
    list(FILTER SOURCES EXCLUDE REGEX ov_remote_context_test.cpp)
endif()

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino_c openvino::util
    commonTestUtils gtest_main)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${OPENVINO_API_SOURCE_DIR}/include>)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
    DATA_PATH=\"${DATA_PATH}\"
    MODELS_PATH=\"${MODELS_PATH}\")

if(TARGET OpenCL::OpenCL)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    add_dependencies(${TARGET_NAME} openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    add_dependencies(${TARGET_NAME} openvino_auto_batch_plugin)
endif()

if(ENABLE_INTEL_CPU)
    add_dependencies(${TARGET_NAME} openvino_intel_cpu_plugin)
endif()

if(ENABLE_INTEL_GPU)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin)
endif()

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION tests
    COMPONENT tests
    EXCLUDE_FROM_ALL)
