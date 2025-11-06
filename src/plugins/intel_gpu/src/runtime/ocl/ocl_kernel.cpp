// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_kernel.hpp"
#include <vector>

namespace cldnn {
namespace ocl {

std::vector<uint8_t> ocl_kernel::get_binary() const {
    // Get the corresponding program object for the kernel
    cl_program program;
    cl_int error = clGetKernelInfo(_compiled_kernel.get(), CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_KERNEL_PROGRAM: " + std::to_string(error));
    }

    // Get the size of the program binary in bytes.
    size_t binary_size = 0;
    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARY_SIZES: " + std::to_string(error));
    }

    // Binary is not available for the device.
    if (binary_size == 0)
        throw std::runtime_error("get_binary: Binary size is zero");

    // Get program binary.
    std::vector<uint8_t> binary(binary_size);
    uint8_t* binary_buffer = binary.data();
    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &binary_buffer, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARIES: " + std::to_string(error));
    }

    return binary;
}

std::string ocl_kernel::get_build_log() const {
    auto program = _compiled_kernel.getInfo<CL_KERNEL_PROGRAM>();
    auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
    // Assume program was build for only 1 device
    // Return first log
    if (log.size() > 0) {
        return log[0].second;
    }
    OPENVINO_THROW("[GPU] Failed to retrieve kernel build log");
}

}  // namespace ocl
}  // namespace cldnn
