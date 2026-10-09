# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(gguf_extensions LANGUAGES CXX)
find_package(OpenVINO REQUIRED)
if(NOT TARGET openvino::frontend::gguf)
    message(FATAL_ERROR "The OpenVINO development package must include the GGUF frontend")
endif()

add_library(gguf_projector_extension MODULE projector.cpp extension.cpp)
add_library(gguf_mmproj_extension MODULE mmproj_extension.cpp)
add_library(gguf_decoder_extension MODULE decoder_extension.cpp)
foreach(extension IN ITEMS gguf_projector_extension gguf_mmproj_extension gguf_decoder_extension)
    target_compile_features(${extension} PRIVATE cxx_std_17)
    target_link_libraries(${extension} PRIVATE openvino::frontend::gguf)
endforeach()

add_executable(gguf_extension_runner runner.cpp)
target_compile_features(gguf_extension_runner PRIVATE cxx_std_17)
target_link_libraries(gguf_extension_runner PRIVATE openvino::runtime openvino::frontend::gguf)
