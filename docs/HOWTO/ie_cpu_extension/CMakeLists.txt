#
# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for building library with user generated CPU extensions
#
# Contains implementation of the basic layer methods
#
# Refer to the section "Adding Your Own Kernels to the Inference Engine" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

# [fft_cmake_list:cmake]
set(CPU_EXTENSIONS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

set(CMAKE_CXX_STANDARD 11)

find_package(ngraph REQUIRED OPTIONAL_COMPONENTS onnx_importer)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)

set(TARGET_NAME fft_cpu_extension)

file(GLOB SRC ${CPU_EXTENSIONS_BASE_DIR}/*.cpp)

add_library(${TARGET_NAME} SHARED ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE ${InferenceEngine_LIBRARIES}
                                             ${NGRAPH_LIBRARIES}
                                             opencv_core)
# [fft_cmake_list:cmake]

