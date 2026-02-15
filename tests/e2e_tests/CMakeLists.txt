# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(e2e_tests)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
