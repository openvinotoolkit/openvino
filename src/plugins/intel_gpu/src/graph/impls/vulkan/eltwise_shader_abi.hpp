// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace cldnn::vulkan::eltwise_shader_abi {

enum class limit : uint32_t {
#define ELTWISE_SHADER_LIMIT(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_LIMIT
};

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

enum class fused_metadata_field : size_t {
#define ELTWISE_SHADER_FUSED_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_METADATA_FIELD
};

enum class fused_chain_metadata_field : size_t {
#define ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_CHAIN_METADATA_FIELD
};

enum class dense_metadata_field : size_t {
#define ELTWISE_SHADER_DENSE_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_DENSE_METADATA_FIELD
};

enum class fused_dense_metadata_field : size_t {
#define ELTWISE_SHADER_FUSED_DENSE_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_DENSE_METADATA_FIELD
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

enum class fused_input_position : uint32_t {
#define ELTWISE_SHADER_FUSED_INPUT_POSITION(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_FUSED_INPUT_POSITION
};

enum class specialization_id : uint32_t {
#define ELTWISE_SHADER_SPECIALIZATION_ID(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_SPECIALIZATION_ID
};

constexpr uint32_t value(mode value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(limit value) {
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

constexpr uint32_t value(fused_input_position value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t index(specialization_id id) {
    return static_cast<uint32_t>(id);
}

constexpr size_t index(metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr size_t index(fused_metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr size_t index(fused_chain_metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr size_t index(dense_metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr size_t index(fused_dense_metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr uint32_t index(tensor_index tensor) {
    return static_cast<uint32_t>(tensor);
}

constexpr uint32_t all_infinities_mask = value(infinity_flag::negative) | value(infinity_flag::positive);

}  // namespace cldnn::vulkan::eltwise_shader_abi
