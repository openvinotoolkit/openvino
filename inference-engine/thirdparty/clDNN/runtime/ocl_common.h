/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>

#include <vector>

namespace cldnn {
namespace gpu {

typedef cl::vector<cl::vector<unsigned char>> kernels_binaries_vector;
typedef cl::vector<kernels_binaries_vector> kernels_binaries_container;
typedef CL_API_ENTRY cl_command_queue(CL_API_CALL* pfn_clCreateCommandQueueWithPropertiesINTEL)(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcodeRet);

using queue_type = cl::CommandQueueIntel;
using kernel_type = cl::KernelIntel;
using kernel_id = std::string;

}  // namespace gpu
}  // namespace cldnn
