# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

set (CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type")
set (CMAKE_CXX_STANDARD 11)

project(time_tests)

set(OpenVINO_MAIN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

add_subdirectory(src)
