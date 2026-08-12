// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace cldnn::vulkan::eltwise_shader_abi {

enum class mode : uint32_t {
#define ELTWISE_SHADER_MODE(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_MODE
};

enum class scalar_type : uint32_t {
#define ELTWISE_SHADER_SCALAR_TYPE(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_SCALAR_TYPE
};

enum class metadata_field : size_t {
#define ELTWISE_SHADER_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_METADATA_FIELD
};

enum class tensor_index : uint32_t {
#define ELTWISE_SHADER_TENSOR_INDEX(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_TENSOR_INDEX
};

enum class storage_flag : uint32_t {
#define ELTWISE_SHADER_STORAGE_FLAG(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_STORAGE_FLAG
};

enum class infinity_flag : uint32_t {
#define ELTWISE_SHADER_INFINITY_FLAG(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_INFINITY_FLAG
};

constexpr uint32_t value(mode value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(scalar_type value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(storage_flag value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(infinity_flag value) {
    return static_cast<uint32_t>(value);
}

constexpr size_t index(metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr uint32_t index(tensor_index tensor) {
    return static_cast<uint32_t>(tensor);
}

constexpr uint32_t all_infinities_mask = value(infinity_flag::negative) | value(infinity_flag::positive);

}  // namespace cldnn::vulkan::eltwise_shader_abi
