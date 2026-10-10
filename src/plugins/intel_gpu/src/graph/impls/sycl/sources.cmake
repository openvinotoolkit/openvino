# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Headers are always compiled in, regardless of which .cpp below is selected.
set(GPU_SYCL_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/eltwise.hpp
    ${CMAKE_CURRENT_LIST_DIR}/impl_example.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mem_adapter.hpp
    ${CMAKE_CURRENT_LIST_DIR}/primitive_sycl_base.h
)

# eltwise.cpp is the real SYCL implementation (OV_GPU_RT STREQUAL "SYCL");
# impl_example.cpp is a placeholder used for all other runtimes.
set(GPU_SYCL_ELTWISE_SOURCE ${CMAKE_CURRENT_LIST_DIR}/eltwise.cpp)
set(GPU_SYCL_IMPL_EXAMPLE_SOURCE ${CMAKE_CURRENT_LIST_DIR}/impl_example.cpp)
