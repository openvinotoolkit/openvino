# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if((CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG) AND CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    set(CMAKE_CXX_FLAGS "-Wno-error=deprecated ${CMAKE_CXX_FLAGS}")
endif()

ov_add_frontend(NAME jax
                LINKABLE_FRONTEND
                DISABLE_CPP_INSTALL
                FILEDESCRIPTION "FrontEnd to load and convert JAX/Flax models"
                LINK_LIBRARIES openvino::util openvino::core::dev)
