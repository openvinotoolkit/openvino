# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(GPU_CPU_IMPL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/activation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/assign.cpp
    ${CMAKE_CURRENT_LIST_DIR}/broadcast.cpp
    ${CMAKE_CURRENT_LIST_DIR}/concat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_impl_helpers.hpp
    ${CMAKE_CURRENT_LIST_DIR}/crop.cpp
    ${CMAKE_CURRENT_LIST_DIR}/detection_output.cpp
    ${CMAKE_CURRENT_LIST_DIR}/eltwise.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fake_convert.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gather.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_mask_gen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/non_max_suppression.cpp
    ${CMAKE_CURRENT_LIST_DIR}/proposal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/range.cpp
    ${CMAKE_CURRENT_LIST_DIR}/read_value.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reduce.cpp
    ${CMAKE_CURRENT_LIST_DIR}/register.cpp
    ${CMAKE_CURRENT_LIST_DIR}/register.hpp
    ${CMAKE_CURRENT_LIST_DIR}/reorder.cpp
    ${CMAKE_CURRENT_LIST_DIR}/scatter_update.cpp
    ${CMAKE_CURRENT_LIST_DIR}/select.cpp
    ${CMAKE_CURRENT_LIST_DIR}/shape_of.cpp
    ${CMAKE_CURRENT_LIST_DIR}/strided_slice.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tile.cpp
)
