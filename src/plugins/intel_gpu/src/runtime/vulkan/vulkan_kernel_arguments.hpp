// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "vulkan_memory.hpp"

namespace cldnn::vulkan {

struct vulkan_prepared_arguments final {
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<vulkan_buffer_allocation::ptr> allocations;
    std::vector<VkAccessFlags2> accesses;
    std::vector<memory::cptr> memories;
    std::vector<uint8_t> push_constants;
};

vulkan_prepared_arguments prepare_vulkan_arguments(const kernel_arguments_desc& descriptor, const kernel_arguments_data& data);

}  // namespace cldnn::vulkan
