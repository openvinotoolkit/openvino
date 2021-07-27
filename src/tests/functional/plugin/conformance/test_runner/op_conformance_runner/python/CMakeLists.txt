# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project (py_conf_runner)

pybind11_add_module(py_conf_runner MODULE py_runner.cpp ../src/core_config.cpp ../../conformance_infra/src/skip_tests_config.cpp ../../conformance_infra/include/conformance.hpp)

target_link_libraries(${PROJECT_NAME} PRIVATE funcSharedTests)
include_directories(../include)