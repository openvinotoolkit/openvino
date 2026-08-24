// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../eltwise_shader_abi.hpp"

namespace cldnn {

struct fused_primitive_desc;
struct layout;
struct program_node;

namespace vulkan::eltwise_detail {

struct fused_eltwise_info {
    const fused_primitive_desc* descriptor = nullptr;
    size_t external_dependency_index = 0;
    eltwise_shader_abi::fused_input_position external_position = eltwise_shader_abi::fused_input_position::rhs;
    bool broadcast_input = false;
};

using fused_eltwise_chain = std::vector<fused_eltwise_info>;

struct fused_post_op_info {
    const fused_primitive_desc* descriptor = nullptr;
    eltwise_shader_abi::post_op_kind kind = eltwise_shader_abi::post_op_kind::activation;
    eltwise_shader_abi::post_activation activation = eltwise_shader_abi::post_activation::none;
    uint32_t quantize_flags = 0;
};

std::optional<fused_post_op_info> get_supported_fused_post_op(const program_node& node);
bool can_use_fused_post_op_kernel(const layout& input0_layout, const layout& input1_layout, const fused_post_op_info& post_op, const layout& output_layout);
bool is_numpy_broadcast_compatible(const layout& input_layout, const layout& output_layout);
std::optional<fused_eltwise_chain> get_supported_fused_eltwise_chain(const program_node& node);

}  // namespace vulkan::eltwise_detail
}  // namespace cldnn
