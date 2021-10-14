# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    find_package(InferenceEngineDeveloperPackage REQUIRED)
    find_package(OpenVINO REQUIRED)
endif()

add_subdirectory(compatibility)
add_subdirectory(pyopenvino)
