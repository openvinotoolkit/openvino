# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.18)
project(ov_node_addon)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DNAPI_VERSION=6)

set(dist_folder "${CMAKE_SOURCE_DIR}/bin/")
set(ov_runtime_folder "$ENV{OV_RUNTIME_DIR}")

if (NOT DEFINED ENV{OV_RUNTIME_DIR})
    message(FATAL_ERROR "You must set OV_RUNTIME_DIR environment variable")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/node-api-headers/include)

include_directories(${ov_runtime_folder}/include)
include_directories(${ov_runtime_folder}/include/ie)

# Create a library
add_library(${PROJECT_NAME} SHARED
    src/node_output.cpp
    src/async_reader.cpp
    src/preprocess/preprocess.cpp
    src/preprocess/pre_post_process_wrap.cpp
    src/preprocess/preprocess_steps.cpp
    src/preprocess/input_info.cpp
    src/preprocess/output_info.cpp
    src/preprocess/input_tensor_info.cpp
    src/preprocess/output_tensor_info.cpp
    src/preprocess/input_model_info.cpp
    src/preprocess/resize_algorithm.cpp
    src/errors.cpp
    src/helper.cpp
    src/tensor.cpp
    src/infer_request.cpp
    src/compiled_model.cpp
    src/core_wrap.cpp
    src/model_wrap.cpp
    src/addon.cpp
    src/element_type.cpp
    src/partial_shape_wrap.cpp
)
target_link_directories(${PROJECT_NAME} PRIVATE ${ov_runtime_folder}/lib/intel64)
target_link_libraries(${PROJECT_NAME} PRIVATE ${ov_runtime_folder}/lib/intel64/libopenvino.so.2023.3.0)
target_link_libraries(${PROJECT_NAME} PRIVATE ${ov_runtime_folder}/lib/intel64/libopenvino_intel_cpu_plugin.so)
target_link_libraries(${PROJECT_NAME} PRIVATE ${ov_runtime_folder}/lib/intel64/libopenvino_ir_frontend.so.2023.3.0)

# Set library properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN/"
    PREFIX ""
    SUFFIX ".node"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE)

# Specify include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/node-addon-api
    include
)

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${dist_folder}

    COMMAND ${CMAKE_COMMAND} -E copy
        ${ov_runtime_folder}/lib/intel64/libopenvino.so.2023.3.0
        ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ov_runtime_folder}/lib/intel64/libopenvino.so.2330
        ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ov_runtime_folder}/lib/intel64/libopenvino_intel_cpu_plugin.so
        ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ov_runtime_folder}/lib/intel64/libopenvino_ir_frontend.so.2023.3.0
        ${dist_folder}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ov_runtime_folder}/lib/intel64/libopenvino_ir_frontend.so.2330
        ${dist_folder}
  )
