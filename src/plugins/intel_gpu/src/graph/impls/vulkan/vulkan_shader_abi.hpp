// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace cldnn::vulkan::shader_abi {

/// Scalar element types shared by Vulkan operation shaders.
enum class scalar_type : uint32_t {
#define VULKAN_SHADER_SCALAR_TYPE(name, value) name = value,
#include "vulkan_shader_abi.inc"
#undef VULKAN_SHADER_SCALAR_TYPE
};

/// Common specialization identifiers used by Vulkan shaders in this backend.
enum class specialization_id : uint32_t {
#define VULKAN_SHADER_SPECIALIZATION_ID(name, value) name = value,
#include "vulkan_shader_abi.inc"
#undef VULKAN_SHADER_SPECIALIZATION_ID
};

constexpr uint32_t index(specialization_id id) {
    return static_cast<uint32_t>(id);
}

constexpr uint32_t value(scalar_type type) {
    return static_cast<uint32_t>(type);
}

}  // namespace cldnn::vulkan::shader_abi
