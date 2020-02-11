// Copyright (C) 2018-2020 Intel Corporation
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
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#if defined __GNUC__
# pragma GCC system_header
#endif

#include <CL/cl2.hpp>
