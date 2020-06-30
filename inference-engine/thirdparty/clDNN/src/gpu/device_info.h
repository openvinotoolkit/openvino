// Copyright (c) 2016 Intel Corporation
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

#pragma once
#include <cstdint>
#include <memory>
#include "api/device.hpp"
#include <string>

namespace cl {
class Device;
}
namespace cldnn {
namespace gpu {

struct device_info_internal : cldnn::device_info {
    std::uint32_t compute_units_count;
    uint32_t dev_type;
    uint32_t vendor_id;
    uint8_t supports_usm;
    bool supports_optimization_hints;
    bool supports_local_block_io;

    explicit device_info_internal(const cl::Device& device);

    device_info convert_to_api() {
        return { cores_count,
         core_frequency,
         max_work_group_size,
         max_local_mem_size,
         max_global_mem_size,
         max_alloc_mem_size,
         max_image2d_width,
         max_image2d_height,
         supports_fp16,
         supports_fp16_denorms,
         supports_subgroups_short,
         supports_image,
         supports_imad,
         supports_immad,
         supports_usm,
         dev_name,
         driver_version
        };
    }
};

}  // namespace gpu
}  // namespace cldnn
