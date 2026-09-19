// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vulkan_memory.hpp"

namespace cldnn::vulkan {
namespace detail {

void check_vulkan_memory_result(VkResult result, const char* operation);
bool transition_foreign_buffer_ownership(const std::shared_ptr<vulkan_device>& device_owner, VkBuffer buffer, bool acquire) noexcept;
vulkan_buffer_allocation::ptr allocate_vulkan_buffer(vulkan_engine& engine, size_t requested_size, vulkan_buffer_memory_usage usage);

}  // namespace detail
}  // namespace cldnn::vulkan
