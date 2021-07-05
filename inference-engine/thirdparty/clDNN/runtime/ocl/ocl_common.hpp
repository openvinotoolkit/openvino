// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ocl_wrapper.hpp"

#include <vector>

namespace cldnn {
namespace ocl {

typedef cl::vector<cl::vector<unsigned char>> kernels_binaries_vector;
typedef cl::vector<kernels_binaries_vector> kernels_binaries_container;
typedef CL_API_ENTRY cl_command_queue(CL_API_CALL* pfn_clCreateCommandQueueWithPropertiesINTEL)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcodeRet);

using ocl_queue_type = cl::CommandQueue;
using ocl_kernel_type = cl::KernelIntel;

class ocl_error : public std::runtime_error {
public:
    explicit ocl_error(cl::Error const& err);
};

}  // namespace ocl
}  // namespace cldnn
