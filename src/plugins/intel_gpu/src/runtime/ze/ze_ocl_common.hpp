// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "ocl/ocl_wrapper.hpp"

#include <string>
#include <exception>

// Prints warning if OpenCL command does not return success result
#define OV_OCL_WARN(f) \
    do { \
        cl_int res_ = (f); \
        if (res_ != CL_SUCCESS) { \
            GPU_DEBUG_INFO << "[Warning] [GPU] " #f " command failed with code " \
                << std::to_string(res_); \
        } \
    } while (false)
