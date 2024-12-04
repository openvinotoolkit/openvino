// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

namespace tests {

bool is_supported_sdpa_micro_kernel(const char* device_id) {
    // Get list of OpenCL platforms.
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);

    // Get first available GPU device
    std::vector<cl::Device> device;
    for (auto p = platform.begin(); p != platform.end(); p++) {
        std::vector<cl::Device> pldev;
        try {
            p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);
            for (auto d = pldev.begin(); d != pldev.end(); d++) {
                if (d->getInfo<CL_DEVICE_AVAILABLE>())
                    device.push_back(*d);
            }
        } catch (...) {
            device.clear();
        }
    }

    if (device.size() == 0)
        return false;

    size_t device_idx = 0;
    if (strlen(device_id) == 5) {
        device_idx = std::atoi(&device_id[4]);
    }

    std::vector<cl::Device> tgt_device{device[device_idx]};
    cl::Context __context = cl::Context(tgt_device);

    // This program check that all required vISA features are supported by current IGC version
    const char* kernel_code = R""""(
        kernel void igc_check() {
            __asm__ volatile(
                    ".decl AA0 v_type=G type=ud num_elts=1\n"
                    ".decl AA1 v_type=G type=ud num_elts=1\n"
                    ".implicit_PSEUDO_INPUT AA0 offset=256 size=4\n"
                    ".implicit_PSEUDO_INPUT AA1 offset=256 size=4\n"
                    "mov (M1_NM,1) AA0(0,0)<1> AA1(0,0)<0;1,0>\n"
            );
        }
        )"""";

    cl::Program program(__context, std::string(kernel_code));

    try
    {
        program.build(device[device_idx]);
    }
    catch (const cl::Error &)
    {
        return false;
    }
    return true;
}

} // namespace tests