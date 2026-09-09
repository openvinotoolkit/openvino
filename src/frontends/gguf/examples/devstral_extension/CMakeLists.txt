# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(gguf_devstral_extension LANGUAGES CXX)
find_package(OpenVINO REQUIRED)
add_library(gguf_devstral_extension MODULE devstral.cpp extension.cpp)
target_compile_features(gguf_devstral_extension PRIVATE cxx_std_17)
target_link_libraries(gguf_devstral_extension PRIVATE openvino::frontend::gguf)
