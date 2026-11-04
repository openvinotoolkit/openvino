# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GPU_COMMON_IMPL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/condition.cpp
    ${CMAKE_CURRENT_LIST_DIR}/loop.cpp
    ${CMAKE_CURRENT_LIST_DIR}/loop.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mlir_primitive.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mlir_primitive.hpp
    ${CMAKE_CURRENT_LIST_DIR}/register.cpp
    ${CMAKE_CURRENT_LIST_DIR}/register.hpp
    ${CMAKE_CURRENT_LIST_DIR}/wait_for_events.cpp
)
