// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"

#include <CL/cl.h>
#include <string>
#include <exception>

// Expect success of OpenCL command, throw runtime error otherwise.
// if already handling an exception (stack unwinding), logging a warning instead to avoid termination.
#define OV_OCL_EXPECT(f)                                                        \
    do {                                                                        \
        cl_int res_ = (f);                                                      \
        if (res_ != CL_SUCCESS) {                                               \
            if (std::uncaught_exceptions() > 0) {                               \
                GPU_DEBUG_INFO << ("[GPU] " #f " command failed with code "     \
                    << std::to_string(res_)                                     \
                    << " (during stack unwinding)");                            \
            } else {                                                            \
                OPENVINO_THROW(#f " command failed with code ", std::to_string(res_)); \
            }                                                                   \
        }                                                                       \
    } while (false)

// Prints warning if OpenCL command does not return success result
#define OV_OCL_WARN(f) \
    do { \
        cl_int res_ = (f); \
        if (res_ != CL_SUCCESS) { \
            GPU_DEBUG_INFO << "[Warning] [GPU] " #f " command failed with code " \
                << std::to_string(res_); \
        } \
    } while (false)
