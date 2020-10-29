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

    supports_imad = true;
    supports_immad = false;

    dev_type = static_cast<uint32_t>(device.getInfo<CL_DEVICE_TYPE>());
    vendor_id = static_cast<uint32_t>(device.getInfo<CL_DEVICE_VENDOR_ID>());

    supports_usm = extensions.find("cl_intel_unified_shared_memory") != std::string::npos;
}
}  // namespace gpu
}  // namespace cldnn
