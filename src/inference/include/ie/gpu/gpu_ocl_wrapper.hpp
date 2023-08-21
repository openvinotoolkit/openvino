// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * OpenCL context and OpenCL shared memory blobs
 *
 * @file gpu_context_api_ocl.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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

#ifdef OV_GPU_USE_OPENCL_HPP
#    include <CL/opencl.hpp>
#else
#    include <CL/cl2.hpp>
#endif

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
