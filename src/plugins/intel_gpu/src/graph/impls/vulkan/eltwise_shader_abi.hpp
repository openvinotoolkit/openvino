// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "vulkan_shader_abi.hpp"

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

using scalar_type = shader_abi::scalar_type;

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

enum class post_op_kind : uint32_t {
#define ELTWISE_SHADER_POST_OP_KIND(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_POST_OP_KIND
};

enum class post_activation : uint32_t {
#define ELTWISE_SHADER_POST_ACTIVATION(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_POST_ACTIVATION
};

enum class post_quantize_flag : uint32_t {
#define ELTWISE_SHADER_POST_QUANTIZE_FLAG(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_POST_QUANTIZE_FLAG
};

enum class post_op_metadata_field : size_t {
#define ELTWISE_SHADER_POST_OP_METADATA_FIELD(name, code) name = code,
#include "eltwise_shader_abi.inc"
#undef ELTWISE_SHADER_POST_OP_METADATA_FIELD
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
    return shader_abi::value(value);
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

constexpr uint32_t value(post_op_kind value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(post_activation value) {
    return static_cast<uint32_t>(value);
}

constexpr uint32_t value(post_quantize_flag value) {
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

constexpr size_t index(post_op_metadata_field field) {
    return static_cast<size_t>(field);
}

constexpr uint32_t index(tensor_index tensor) {
    return static_cast<uint32_t>(tensor);
}

constexpr uint32_t all_infinities_mask = value(infinity_flag::negative) | value(infinity_flag::positive);

constexpr bool post_op_fusion_has_enough_work(uint64_t element_count, uint64_t max_work_group_size, uint64_t subgroup_size) {
    const auto subgroup_lane_granularity = static_cast<uint64_t>(value(limit::post_op_fusion_subgroup_lane_granularity));
    const auto stage_count = static_cast<uint64_t>(value(limit::post_op_fusion_stage_count));
    const auto normalized_subgroup_size = subgroup_size == 0 ? subgroup_lane_granularity : subgroup_size;
    const auto subgroup_scale = (normalized_subgroup_size - 1) / subgroup_lane_granularity + 1;
    const auto normalized_work_group_size = max_work_group_size == 0 ? uint64_t{1} : max_work_group_size;
    if (subgroup_scale > std::numeric_limits<uint64_t>::max() / stage_count ||
        normalized_work_group_size > std::numeric_limits<uint64_t>::max() / (subgroup_scale * stage_count)) {
        return false;
    }
    return element_count >= normalized_work_group_size * subgroup_scale * stage_count;
}

}  // namespace cldnn::vulkan::eltwise_shader_abi
