// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vk_engine.hpp"

#include <memory>

namespace ov::core::vulkan {
namespace cross_platform {

// Entry point: creates the Vulkan engine for the requested platform
// (vk_platform_config; defaults come from the environment, see vk_platform.hpp).
inline std::shared_ptr<vk_engine> create_vk_engine(const vk_platform_config& config = platform_config_from_env()) {
    return vk_engine::create(config);
}

}  // namespace cross_platform
}  // namespace ov::core::vulkan
