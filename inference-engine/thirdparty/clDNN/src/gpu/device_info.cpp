// Copyright (c) 2018 Intel Corporation
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

#include "device_info.h"
#include "include/to_string_utils.h"
#include <unordered_map>
#include <string>
#include <cassert>
#include <time.h>
#include <limits>
#include <chrono>
#include "ocl_builder.h"

#include <fstream>
#include <iostream>
#include <utility>


namespace cldnn {
namespace gpu {

namespace {

bool is_local_block_io_supported(const cl::Device& device) {
    try {
        cl::Context ctx(device);
        std::string kernel_code =
            "__attribute__((intel_reqd_sub_group_size(8)))"
            "__attribute__((reqd_work_group_size(8, 1, 1)))"
            "void kernel is_local_block_io_supported(global uchar* dst) {"
            "    uint lid = get_sub_group_local_id();"
            "    uchar val = (uchar)lid * 2;"
            "    __local uchar tmp_slm[8];"
            "    intel_sub_group_block_write_uc2(tmp_slm, (uchar2)(val));"
            "    barrier(CLK_LOCAL_MEM_FENCE);"
            "    uchar2 read = intel_sub_group_block_read_uc2(tmp_slm);"
            "    dst[lid] = read.s0 + 1;"
            "}";
        cl::Program program(ctx, kernel_code);
        if (program.build({ device }, "-Dcl_intel_subgroup_local_block_io") != CL_SUCCESS)
            return false;
        cl::Buffer buffer(ctx, CL_MEM_READ_WRITE, sizeof(uint8_t) * 8);
        cl::Kernel kernel(program, "is_local_block_io_supported");
        kernel.setArg(0, buffer);

        cl::Event ev;
        cl::CommandQueue queue(ctx, device);
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(), cl::NDRange(8), cl::NDRange(8), nullptr, &ev);
        ev.wait();

        uint8_t result[8];
        uint8_t expected[8] = { 1, 3, 5, 7, 9, 11, 13, 15 };
        queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(uint8_t) * 8, &result);
        for (int i = 0; i < 8; ++i) {
            if (result[i] != expected[i])
                return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

device_info_internal::device_info_internal(const cl::Device& device) {
    dev_name = device.getInfo<CL_DEVICE_NAME>();
    driver_version = device.getInfo<CL_DRIVER_VERSION>();

    compute_units_count = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    cores_count = static_cast<uint32_t>(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    core_frequency = static_cast<uint32_t>(device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());

    max_work_group_size = static_cast<uint64_t>(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

    if (max_work_group_size > 256)
        max_work_group_size = 256;

    max_local_mem_size = static_cast<uint64_t>(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    max_global_mem_size = static_cast<uint64_t>(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    max_alloc_mem_size = static_cast<uint64_t>(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());

    supports_image = static_cast<uint8_t>(device.getInfo<CL_DEVICE_IMAGE_SUPPORT>());
    max_image2d_width = static_cast<uint64_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>());
    max_image2d_height = static_cast<uint64_t>(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>());

    // Check for supported features.
    auto extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
    extensions.push_back(' ');  // Add trailing space to ease searching (search with keyword with trailing space).

    supports_fp16 = extensions.find("cl_khr_fp16 ") != std::string::npos;
    supports_fp16_denorms = supports_fp16 && (device.getInfo<CL_DEVICE_HALF_FP_CONFIG>() & CL_FP_DENORM) != 0;

    supports_subgroups_short = extensions.find("cl_intel_subgroups_short") != std::string::npos;

    supports_imad = dev_name.find("Gen12") != std::string::npos;
    supports_immad = false;

    dev_type = static_cast<uint32_t>(device.getInfo<CL_DEVICE_TYPE>());
    vendor_id = static_cast<uint32_t>(device.getInfo<CL_DEVICE_VENDOR_ID>());

    supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;

    supports_optimization_hints = false;
    supports_local_block_io = extensions.find("cl_intel_subgroup_local_block_io") != std::string::npos &&
                              is_local_block_io_supported(device);
}
}  // namespace gpu
}  // namespace cldnn
