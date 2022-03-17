// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory blobs
 *
 * @file gpu_context_api_ocl.hpp
 */
#pragma once

/**
 * @brief Definitions required by Khronos headers
 */

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#    define CL_HPP_ENABLE_EXCEPTIONS
#endif

#ifdef CL_HPP_MINIMUM_OPENCL_VERSION
#    if CL_HPP_MINIMUM_OPENCL_VERSION < 120
#        error "CL_HPP_MINIMUM_OPENCL_VERSION must be >= 120"
#    endif
#else
#    define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#    if CL_HPP_TARGET_OPENCL_VERSION < 120
#        error "CL_HPP_TARGET_OPENCL_VERSION must be >= 120"
#    endif
#else
#    define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC system_header
#endif

#include <CL/cl2.hpp>

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
