# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(samples_tests)

# Search OpenVINO Inference Engine installed
find_package(OpenVINO REQUIRED)

install(DIRECTORY smoke_tests/ DESTINATION tests/smoke_tests COMPONENT tests EXCLUDE_FROM_ALL)
