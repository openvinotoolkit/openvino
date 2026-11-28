// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime.h"

#include <CL/cl_ext.h>

#include <atomic>
#include <cstdio>
#include <vector>

namespace {
constexpr const char* kernel_source = R"CLC(
__kernel void add_relu(__global const float* left, __global const float* right, __global float* output) {
    size_t index = get_global_id(0);
    output[index] = fmax(left[index] + right[index], 0.0f);
}
)CLC";
}

namespace mk {
TestRuntime::~TestRuntime() {
    Destroy();
}

TErrorcode TestRuntime::Init(const IConstantParams*, const IPlatformParams* params) {
    const auto* platform_params = static_cast<const mk::OCLPlatformParams*>(params);
    queue = platform_params->stream;
    cl_platform_id platform = nullptr;
    cl_int status = clGetDeviceInfo(platform_params->deviceId,
                                    CL_DEVICE_PLATFORM,
                                    sizeof(platform),
                                    &platform,
                                    nullptr);
    if (status != CL_SUCCESS)
        return status;
    set_pointer = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL"));
    if (!set_pointer)
        return CL_INVALID_OPERATION;
    const char* source = kernel_source;
    program = clCreateProgramWithSource(platform_params->context, 1, &source, nullptr, &status);
    if (status != CL_SUCCESS)
        return status;
    status = clBuildProgram(program, 1, &platform_params->deviceId, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t length = 0;
        clGetProgramBuildInfo(program, platform_params->deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
        std::vector<char> log(length + 1, 0);
        clGetProgramBuildInfo(program, platform_params->deviceId, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
        std::fprintf(stderr, "Test megakernel OpenCL build failed: %s\n", log.data());
        return status;
    }
    kernel = clCreateKernel(program, "add_relu", &status);
    return status;
}

TErrorcode TestRuntime::Execute(const IRuntimeParams* params) {
    const auto* io = static_cast<const RuntimeParamsImpl*>(params);
    cl_int status = set_pointer(kernel, 0, io->left);
    if (status != CL_SUCCESS)
        return status;
    status = set_pointer(kernel, 1, io->right);
    if (status != CL_SUCCESS)
        return status;
    status = set_pointer(kernel, 2, io->output);
    if (status != CL_SUCCESS)
        return status;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &io->count, nullptr, 0, nullptr, nullptr);
    return status;
}

TErrorcode TestRuntime::Destroy() {
    if (kernel) {
        clReleaseKernel(kernel);
        kernel = nullptr;
    }
    if (program) {
        clReleaseProgram(program);
        program = nullptr;
    }
    return CL_SUCCESS;
}
}