// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vulkan_memory.hpp"

namespace cldnn::vulkan {
namespace detail {

void check_vulkan_memory_result(VkResult result, const char* operation);
bool transition_foreign_buffer_ownership(const std::shared_ptr<vulkan_device>& device_owner, VkBuffer buffer, bool acquire) noexcept;

}  // namespace detail
}  // namespace cldnn::vulkan
