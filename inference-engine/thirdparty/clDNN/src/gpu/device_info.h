// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    uint32_t vendor_id;
    uint8_t supports_usm;
    bool supports_optimization_hints;
    bool supports_local_block_io;

    explicit device_info_internal(const cl::Device& device);

    device_info convert_to_api() {
        return { cores_count,
         core_frequency,
         max_threads_per_execution_unit,
         max_threads_per_device,
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
         driver_version,
         dev_type,
         gfx_ver,
         device_id,
         num_slices,
         num_sub_slices_per_slice,
         num_eus_per_sub_slice,
         num_threads_per_eu,
        };
    }
};

}  // namespace gpu
}  // namespace cldnn
