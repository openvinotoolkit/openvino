# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(
    "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/XLink"
    "${CMAKE_BINARY_DIR}/thirdparty/movidius/XLink")

add_subdirectory(
    "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc"
    "${CMAKE_BINARY_DIR}/thirdparty/movidius/mvnc")
