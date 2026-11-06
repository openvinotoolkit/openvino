// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"

namespace {
using namespace ov::pass;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v7 = ov::op::v7;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

// Note: intermediate nodes remain unchanged,
// but we need to explicitly call shape inference for them to keep shapes consistency
void validate_nodes(const pattern::PatternValueMap& map, const std::initializer_list<std::shared_ptr<ov::Node>> nodes) {
    for (const auto& node : nodes) {
        map.at(node).get_node_shared_ptr()->validate_and_infer_types();
    }
};

// Build a per-expert MatMul weight pattern
std::shared_ptr<ov::Node> make_expert_weight_pattern(const std::vector<ov::element::Type>& supported_weights_types) {
    if (supported_weights_types.empty()) {
        return pattern::any_input();
    }
    return std::make_shared<pattern::op::CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{3});
}

std::shared_ptr<ov::op::v0::Unsqueeze> introduce_n_experts_dim(const ov::Output<ov::Node>& data) {
    auto zero_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, zero_const);
    ov::copy_runtime_info(data.get_node_shared_ptr(), {unsqueeze, zero_const});
    return unsqueeze;
}

std::shared_ptr<v8::Gather> get_packed_mxfp4_gather(const ov::Output<ov::Node>& indices) {
    auto convert = ov::as_type_ptr<v0::Convert>(indices.get_node_shared_ptr());
    auto concat = convert ? ov::as_type_ptr<v0::Concat>(convert->input_value(0).get_node_shared_ptr()) : nullptr;
    if (!concat || concat->get_input_size() != 2 || concat->get_axis() != 4) {
        return nullptr;
    }
    auto low = ov::as_type_ptr<ov::op::v13::BitwiseAnd>(concat->input_value(0).get_node_shared_ptr());
    auto high = ov::as_type_ptr<ov::op::v15::BitwiseRightShift>(concat->input_value(1).get_node_shared_ptr());
    if (!low || !high) {
        return nullptr;
    }
    auto low_slice = ov::as_type_ptr<v8::Slice>(low->input_value(0).get_node_shared_ptr());
    auto high_slice = ov::as_type_ptr<v8::Slice>(high->input_value(0).get_node_shared_ptr());
    if (!low_slice || !high_slice || low_slice->input_value(0) != high_slice->input_value(0)) {
        return nullptr;
    }
    return ov::as_type_ptr<v8::Gather>(low_slice->input_value(0).get_node_shared_ptr());
}

std::shared_ptr<v8::Gather> get_packed_mxfp4_scale_gather(const ov::Output<ov::Node>& indices) {
    auto convert = ov::as_type_ptr<v0::Convert>(indices.get_node_shared_ptr());
    auto slice = convert ? ov::as_type_ptr<v8::Slice>(convert->input_value(0).get_node_shared_ptr()) : nullptr;
    return slice ? ov::as_type_ptr<v8::Gather>(slice->input_value(0).get_node_shared_ptr()) : nullptr;
}

bool is_mxfp4_lut_gather(const ov::Output<ov::Node>& output, size_t lut_size) {
    auto gather = ov::as_type_ptr<v8::Gather>(output.get_node_shared_ptr());
    auto constant = gather ? ov::as_type_ptr<v0::Constant>(gather->input_value(0).get_node_shared_ptr()) : nullptr;
    return constant && constant->get_shape() == ov::Shape{lut_size};
}

bool unpack_packed_mxfp4_weights(const std::shared_ptr<v8::Gather>& weights_gather,
                                 std::shared_ptr<v0::Constant>& weights,
                                 std::shared_ptr<v0::Constant>& scales) {
    auto weights_source = weights_gather->input_value(0).get_node_shared_ptr();
    if (auto reshape = ov::as_type_ptr<v1::Reshape>(weights_source)) {
        weights_source = reshape->input_value(0).get_node_shared_ptr();
    }
    auto packed_constant = ov::as_type_ptr<v0::Constant>(weights_source);
    if (!packed_constant || packed_constant->get_element_type() != ov::element::u8) {
        return false;
    }

    auto packed_shape = packed_constant->get_shape();
    if (packed_shape.size() == 5 && packed_shape[0] == 1) {
        packed_shape.erase(packed_shape.begin());
    }
    if (packed_shape.size() != 4 || packed_shape[3] != 17) {
        return false;
    }

    const size_t experts = packed_shape[0];
    const size_t output_features = packed_shape[1];
    const size_t groups = packed_shape[2];
    constexpr size_t group_size = 32;
    const size_t input_features = groups * group_size;
    const auto* packed_data = packed_constant->get_data_ptr<uint8_t>();

    ov::Tensor weights_tensor(ov::element::f4e2m1, ov::Shape{experts, output_features, input_features});
    ov::Tensor scales_tensor(ov::element::f32, ov::Shape{experts, output_features, groups});
    auto* weights_data = static_cast<uint8_t*>(weights_tensor.data());
    std::memset(weights_data, 0, weights_tensor.get_byte_size());
    auto* scales_data = scales_tensor.data<float>();

    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t output = 0; output < output_features; ++output) {
            for (size_t group = 0; group < groups; ++group) {
                const size_t src_offset = ((expert * output_features + output) * groups + group) * 17;
                const uint32_t scale_bits = static_cast<uint32_t>(packed_data[src_offset]) << 23;
                float scale = 0.0f;
                if (packed_data[src_offset] == 0) {
                    scale = std::ldexp(1.0f, -127);
                } else if (packed_data[src_offset] == 255) {
                    scale = std::numeric_limits<float>::quiet_NaN();
                } else {
                    std::memcpy(&scale, &scale_bits, sizeof(scale));
                }
                scales_data[(expert * output_features + output) * groups + group] = scale;

                for (size_t k = 0; k < group_size; ++k) {
                    const auto packed_byte = packed_data[src_offset + 1 + (k % 16)];
                    const auto nibble = static_cast<uint8_t>(k < 16 ? packed_byte & 0x0f : packed_byte >> 4);
                    const size_t dst_index =
                        (expert * output_features + output) * input_features + group * group_size + k;
                    weights_data[dst_index / 2] |= static_cast<uint8_t>(nibble << ((dst_index % 2) * 4));
                }
            }
        }
    }

    weights = std::make_shared<v0::Constant>(weights_tensor);
    scales = std::make_shared<v0::Constant>(scales_tensor);
    ov::copy_runtime_info(packed_constant, {weights, scales});
    return true;
}

// --- Shared pattern node containers ---

struct MOE2GEMMPatternNodes {
    std::shared_ptr<ov::Node> experts_input, tile, after_tile_reshape;
    std::shared_ptr<ov::Node> gate_up_matmul, gate_up_add, gate_up_bias;
    std::shared_ptr<ov::Node> slice1, clamp, add1, slice2, minimum1, swish_beta, swish, multiply2;
    std::shared_ptr<ov::Node> down_proj_matmul, down_proj_bias, down_proj_add;
    std::shared_ptr<ov::Node> end_reshape_target_shape, end_reshape;
    std::shared_ptr<ov::Node> router_topk_indices, chosen_experts, scatter_elements_update;
    std::shared_ptr<ov::Node> router_transpose, router_reshape, optional_unsqueeze;
    std::shared_ptr<ov::Node> mul3, reduceSum_keepDims, reduceSum_squeeze, reduceSum_noKeepDims;
    std::shared_ptr<ov::Node> reduce_sum;
};

MOE2GEMMPatternNodes build_2gemm_pattern(const std::vector<ov::element::Type>& supported_weights_types,
                                         const std::vector<ov::element::Type>& supported_activation_types) {
    MOE2GEMMPatternNodes p;

    p.experts_input = pattern::wrap_type<v1::Reshape>({pattern::any_input(), pattern::any_input()},
                                                      pattern::type_matches_any(supported_activation_types));
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()}, pattern::consumers_count(1));
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()}, pattern::consumers_count(1));
    p.gate_up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, make_expert_weight_pattern(supported_weights_types)},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.gate_up_bias = pattern::wrap_const();
    p.gate_up_add = pattern::wrap_type<v1::Add>({p.gate_up_matmul, p.gate_up_bias}, pattern::consumers_count(2));

    // Branch 1: Slice_1 -> Clamp -> Add_1
    p.slice1 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    p.clamp = pattern::wrap_type<v0::Clamp>({p.slice1}, pattern::consumers_count(1));
    p.add1 = pattern::wrap_type<v1::Add>({p.clamp, pattern::wrap_const()}, pattern::consumers_count(1));

    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    p.slice2 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    p.minimum1 = pattern::wrap_type<v1::Minimum>({p.slice2, pattern::wrap_const()}, pattern::consumers_count(1));
    p.swish_beta = pattern::wrap_const();
    p.swish = pattern::wrap_type<v4::Swish>({p.minimum1, p.swish_beta}, pattern::consumers_count(1));

    // Join: Multiply_2
    p.multiply2 = pattern::wrap_type<v1::Multiply>({p.add1, p.swish}, pattern::consumers_count(1));

    // Down projection
    p.down_proj_matmul = pattern::wrap_type<v0::MatMul>(
        {p.multiply2, make_expert_weight_pattern(supported_weights_types)},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.down_proj_bias = pattern::wrap_const();
    p.down_proj_add = pattern::wrap_type<v1::Add>({p.down_proj_matmul, p.down_proj_bias}, pattern::consumers_count(1));
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape =
        pattern::wrap_type<v1::Reshape>({p.down_proj_add, p.end_reshape_target_shape}, pattern::consumers_count(1));

    // Routing weights/mask
    p.router_topk_indices = pattern::any_input();
    p.chosen_experts = pattern::any_input();
    p.scatter_elements_update = pattern::wrap_type<v3::ScatterElementsUpdate, v12::ScatterElementsUpdate>(
        {pattern::any_input(), p.router_topk_indices, p.chosen_experts, pattern::any_input()});

    p.router_transpose = pattern::wrap_type<v1::Transpose>({p.scatter_elements_update, pattern::any_input()});
    p.router_reshape = pattern::wrap_type<v1::Reshape>({p.router_transpose, pattern::any_input()});
    p.optional_unsqueeze = pattern::optional<v0::Unsqueeze>({p.router_reshape, pattern::any_input()});

    p.mul3 = pattern::wrap_type<v1::Multiply>({p.end_reshape, p.optional_unsqueeze});
    // For ARM, Reduce ops are implemented with keep_dims=true followed by Squeeze operation
    p.reduceSum_keepDims = pattern::wrap_type<ov::op::v1::ReduceSum>({p.mul3, pattern::any_input()},
                                                                     pattern::consumers_count(1),
                                                                     {{"keep_dims", true}});
    p.reduceSum_squeeze = pattern::wrap_type<ov::op::v0::Squeeze>({p.reduceSum_keepDims, pattern::any_input()});
    p.reduceSum_noKeepDims = pattern::wrap_type<ov::op::v1::ReduceSum>({p.mul3, pattern::any_input()},
                                                                       pattern::consumers_count(1),
                                                                       {{"keep_dims", false}});
    p.reduce_sum = p.reduceSum_squeeze | p.reduceSum_noKeepDims;

    return p;
}

struct MOE3GEMMPatternNodes {
    std::shared_ptr<ov::Node> experts_input, tile, after_tile_reshape;
    std::shared_ptr<ov::Node> gate_matmul, swish, up_matmul, swiglu;
    std::shared_ptr<ov::Node> down_matmul;
    std::shared_ptr<ov::Node> end_reshape_target_shape, end_reshape;
    std::shared_ptr<ov::Node> router_topk_indices, chosen_experts, scatter_elements_update;
    std::shared_ptr<ov::Node> router_transpose, router_reshape, optional_unsqueeze;
    std::shared_ptr<ov::Node> mul3, reduceSum_keepDims, reduceSum_squeeze, reduceSum_noKeepDims;
    std::shared_ptr<ov::Node> reduce_sum;
};

MOE3GEMMPatternNodes build_3gemm_pattern(const std::vector<ov::element::Type>& supported_weights_types,
                                         const std::vector<ov::element::Type>& supported_activation_types) {
    MOE3GEMMPatternNodes p;

    p.experts_input = pattern::any_input(pattern::type_matches_any(supported_activation_types));
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()}, pattern::consumers_count(1));
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()}, pattern::consumers_count(2));

    // First GEMM (activation gate)
    p.gate_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, make_expert_weight_pattern(supported_weights_types)},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.swish = pattern::wrap_type<v4::Swish, v7::Gelu>({p.gate_matmul}, pattern::consumers_count(1));
    // Second GEMM (up_projection)
    p.up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, make_expert_weight_pattern(supported_weights_types)},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    // Join: Multiply (SwiGLU)
    p.swiglu = pattern::wrap_type<v1::Multiply>({p.swish, p.up_matmul}, pattern::consumers_count(1));

    // Third GEMM (down_projection)
    p.down_matmul = pattern::wrap_type<v0::MatMul>(
        {p.swiglu, make_expert_weight_pattern(supported_weights_types)},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape =
        pattern::wrap_type<v1::Reshape>({p.down_matmul, p.end_reshape_target_shape}, pattern::consumers_count(1));

    // Routing weights/mask
    p.router_topk_indices = pattern::any_input();
    p.chosen_experts = pattern::any_input();
    p.scatter_elements_update = pattern::wrap_type<v3::ScatterElementsUpdate, v12::ScatterElementsUpdate>(
        {pattern::any_input(), p.router_topk_indices, p.chosen_experts, pattern::any_input()});
    p.router_transpose = pattern::wrap_type<v1::Transpose>({p.scatter_elements_update, pattern::any_input()});
    p.router_reshape = pattern::wrap_type<v1::Reshape>({p.router_transpose, pattern::any_input()});
    p.optional_unsqueeze = pattern::optional<v0::Unsqueeze>({p.router_reshape, pattern::any_input()});

    p.mul3 = pattern::wrap_type<v1::Multiply>({p.end_reshape, p.optional_unsqueeze});
    // For ARM, Reduce ops are implemented with keep_dims=true followed by Squeeze operation
    p.reduceSum_keepDims = pattern::wrap_type<ov::op::v1::ReduceSum>({p.mul3, pattern::any_input()},
                                                                     pattern::consumers_count(1),
                                                                     {{"keep_dims", true}});
    p.reduceSum_squeeze = pattern::wrap_type<ov::op::v0::Squeeze>({p.reduceSum_keepDims, pattern::any_input()});
    p.reduceSum_noKeepDims = pattern::wrap_type<ov::op::v1::ReduceSum>({p.mul3, pattern::any_input()},
                                                                       pattern::consumers_count(1),
                                                                       {{"keep_dims", false}});
    p.reduce_sum = p.reduceSum_squeeze | p.reduceSum_noKeepDims;

    return p;
}

}  // namespace

namespace ov::pass {

using ov::op::internal::GatherMatmul;

// ============================================================================
// BGM-producing passes (IR → GatherMatmul)
// ============================================================================

ConvertTiledMoeBlockTo2GatherMatmuls::ConvertTiledMoeBlockTo2GatherMatmuls(
    const std::vector<ov::element::Type>& supported_weights_types,
    const std::vector<ov::element::Type>& supported_activation_types) {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo2GatherMatmuls);

    auto p = build_2gemm_pattern(supported_weights_types, supported_activation_types);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& experts_subgraph_input = pm.at(p.experts_input);
        const auto& active_indices = pm.at(p.router_topk_indices);

        const auto gate_up_mm_node = pm.at(p.gate_up_matmul).get_node_shared_ptr();
        const auto gate_up_add_node = pm.at(p.gate_up_add).get_node_shared_ptr();
        const auto gate_up_bias_node = pm.at(p.gate_up_bias).get_node_shared_ptr();

        // GatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        // Number of activated experts is always 1 for the first GatherMatmul
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);
        const auto gate_up_gathered_mm = std::make_shared<GatherMatmul>(unsqueeze,
                                                                        gate_up_mm_node->input_value(1),
                                                                        active_indices,
                                                                        gate_up_bias_node);
        ov::replace_node_update_name(gate_up_add_node, gate_up_gathered_mm);

        validate_nodes(pm, {p.slice1, p.clamp, p.add1, p.slice2, p.minimum1, p.swish, p.multiply2});

        const auto down_proj_mm_node = pm.at(p.down_proj_matmul).get_node_shared_ptr();
        const auto down_proj_bias_node = pm.at(p.down_proj_bias).get_node_shared_ptr();

        const auto down_gathered_mm = std::make_shared<GatherMatmul>(pm.at(p.multiply2),
                                                                     down_proj_mm_node->input_value(1),
                                                                     active_indices,
                                                                     down_proj_bias_node);
        ov::copy_runtime_info(down_proj_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_proj_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pm.at(p.chosen_experts);
        const auto router_transpose_node = pm.at(p.router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            std::make_shared<v1::Transpose>(chosen_experts_input, router_transpose_node->input_value(1));
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        const auto router_unsqueeze_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze = std::make_shared<v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pm.at(p.mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());

        std::shared_ptr<ov::Node> new_reduce_sum = nullptr;
        std::string new_reduce_sum_friendly_name;
        if (pm.find(p.reduceSum_squeeze) != pm.end()) {
            const auto reduce_node = pm.at(p.reduceSum_keepDims).get_node_shared_ptr();
            const auto squeeze_node = pm.at(p.reduceSum_squeeze).get_node_shared_ptr();
            const auto new_reduce_node =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_node);
            new_reduce_sum =
                squeeze_node->clone_with_new_inputs({new_reduce_node->output(0), squeeze_node->input_value(1)});
            ov::copy_runtime_info(squeeze_node, new_reduce_sum);
            new_reduce_sum_friendly_name = squeeze_node->get_friendly_name();
        } else {
            const auto reduce_node = pm.at(p.reduceSum_noKeepDims).get_node_shared_ptr();
            new_reduce_sum =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_sum);
            new_reduce_sum_friendly_name = reduce_node->get_friendly_name();
        }
        new_reduce_sum->set_friendly_name(new_reduce_sum_friendly_name);

        const auto& end_reshape_out = pm.at(p.end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pm.at(p.end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto shape_slice = std::make_shared<v8::Slice>(
            end_reshape_shape,
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {shape_slice,
                               shape_slice->get_input_node_shared_ptr(1),
                               shape_slice->get_input_node_shared_ptr(2),
                               shape_slice->get_input_node_shared_ptr(3),
                               shape_slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<v1::Reshape>(new_reduce_sum, shape_slice, true);
        ov::replace_output_update_name(pm.at(p.reduce_sum), reshape->output(0));
        // To avoid friendly name duplication
        reshape->set_friendly_name(reshape->get_friendly_name() + "_Reshape");
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(p.reduce_sum, matcher_name);
    this->register_matcher(matcher, callback);
}

ConvertTiledMoeBlockTo3GatherMatmuls::ConvertTiledMoeBlockTo3GatherMatmuls(
    const std::vector<ov::element::Type>& supported_weights_types,
    const std::vector<ov::element::Type>& supported_activation_types) {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo3GatherMatmuls);

    auto p = build_3gemm_pattern(supported_weights_types, supported_activation_types);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& experts_subgraph_input = pm.at(p.experts_input);
        const auto& active_indices = pm.at(p.router_topk_indices);

        const auto gate_mm_node = pm.at(p.gate_matmul).get_node_shared_ptr();
        const auto up_mm_node = pm.at(p.up_matmul).get_node_shared_ptr();
        const auto down_mm_node = pm.at(p.down_matmul).get_node_shared_ptr();

        // GatherMatmul A shape: [n_activated_experts, batch_size * seq_length, hidden_size]
        // Number of activated experts is always 1 for the first GatherMatmul
        const auto unsqueeze = introduce_n_experts_dim(experts_subgraph_input);
        const auto gate_gathered_mm =
            std::make_shared<GatherMatmul>(unsqueeze, gate_mm_node->input_value(1), active_indices);
        const auto up_gathered_mm =
            std::make_shared<GatherMatmul>(unsqueeze, up_mm_node->input_value(1), active_indices);
        ov::replace_node_update_name(gate_mm_node, gate_gathered_mm);
        ov::replace_node_update_name(up_mm_node, up_gathered_mm);

        validate_nodes(pm, {p.swish, p.swiglu});

        const auto down_gathered_mm =
            std::make_shared<GatherMatmul>(pm.at(p.swiglu), down_mm_node->input_value(1), active_indices);
        ov::copy_runtime_info(down_mm_node, down_gathered_mm);
        down_gathered_mm->set_friendly_name(down_mm_node->get_friendly_name());

        const auto& chosen_experts_input = pm.at(p.chosen_experts);
        const auto router_transpose_node = pm.at(p.router_transpose).get_node_shared_ptr();
        const auto new_router_transpose =
            std::make_shared<v1::Transpose>(chosen_experts_input, router_transpose_node->input_value(1));
        ov::copy_runtime_info(router_transpose_node, new_router_transpose);
        new_router_transpose->set_friendly_name(router_transpose_node->get_friendly_name());

        const auto router_unsqueeze_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        const auto router_unsqueeze = std::make_shared<v0::Unsqueeze>(new_router_transpose, router_unsqueeze_const);
        ov::copy_runtime_info(router_transpose_node, {router_unsqueeze_const, router_unsqueeze});

        const auto final_mul_node = pm.at(p.mul3).get_node_shared_ptr();
        const auto new_final_mul =
            final_mul_node->clone_with_new_inputs({down_gathered_mm->output(0), router_unsqueeze->output(0)});
        ov::copy_runtime_info(final_mul_node, new_final_mul);
        new_final_mul->set_friendly_name(final_mul_node->get_friendly_name());

        std::shared_ptr<ov::Node> new_reduce_sum = nullptr;
        std::string new_reduce_sum_friendly_name;
        if (pm.find(p.reduceSum_squeeze) != pm.end()) {
            const auto reduce_node = pm.at(p.reduceSum_keepDims).get_node_shared_ptr();
            const auto squeeze_node = pm.at(p.reduceSum_squeeze).get_node_shared_ptr();
            const auto new_reduce_node =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_node);
            new_reduce_sum =
                squeeze_node->clone_with_new_inputs({new_reduce_node->output(0), squeeze_node->input_value(1)});
            ov::copy_runtime_info(squeeze_node, new_reduce_sum);
            new_reduce_sum_friendly_name = squeeze_node->get_friendly_name();
        } else {
            const auto reduce_node = pm.at(p.reduceSum_noKeepDims).get_node_shared_ptr();
            new_reduce_sum =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_sum);
            new_reduce_sum_friendly_name = reduce_node->get_friendly_name();
        }
        new_reduce_sum->set_friendly_name(new_reduce_sum_friendly_name);

        const auto& end_reshape_out = pm.at(p.end_reshape);
        const auto end_reshape_rank = end_reshape_out.get_partial_shape().rank();
        const auto& end_reshape_shape = pm.at(p.end_reshape_target_shape);
        // n_all_experts dimension is cut off after ReduceSum
        const auto shape_slice = std::make_shared<v8::Slice>(
            end_reshape_shape,
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {end_reshape_rank.get_length()}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
            v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
        ov::copy_runtime_info(end_reshape_out.get_node_shared_ptr(),
                              {shape_slice,
                               shape_slice->get_input_node_shared_ptr(1),
                               shape_slice->get_input_node_shared_ptr(2),
                               shape_slice->get_input_node_shared_ptr(3),
                               shape_slice->get_input_node_shared_ptr(4)});

        const auto reshape = std::make_shared<v1::Reshape>(new_reduce_sum, shape_slice, true);
        ov::replace_output_update_name(pm.at(p.reduce_sum), reshape->output(0));
        reshape->set_friendly_name(new_reduce_sum->get_friendly_name() + "_Reshape");
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(p.reduce_sum, matcher_name);
    this->register_matcher(matcher, callback);
}

ConvertPackedMoeMatMulToGatherMatmul::ConvertPackedMoeMatMulToGatherMatmul() {
    MATCHER_SCOPE(ConvertPackedMoeMatMulToGatherMatmul);

    auto matmul_pattern =
        pattern::wrap_type<v0::MatMul>({pattern::any_input(), pattern::any_input()},
                                       pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto matmul = ov::as_type_ptr<v0::MatMul>(m.get_match_root());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto expanded_activations = ov::as_type_ptr<v0::Unsqueeze>(matmul->input_value(0).get_node_shared_ptr());
        auto expanded_weights = ov::as_type_ptr<v1::Reshape>(matmul->input_value(1).get_node_shared_ptr());
        auto weight_scale_product =
            expanded_weights ? ov::as_type_ptr<v1::Multiply>(expanded_weights->input_value(0).get_node_shared_ptr())
                             : nullptr;
        if (!expanded_activations || !expanded_weights || !weight_scale_product ||
            matmul->get_input_partial_shape(0).rank() != 4 || matmul->get_input_partial_shape(1).rank() != 4 ||
            expanded_activations->input_value(0).get_partial_shape().rank() != 3 ||
            !is_mxfp4_lut_gather(weight_scale_product->input_value(0), 16) ||
            !is_mxfp4_lut_gather(weight_scale_product->input_value(1), 256)) {
            return false;
        }

        auto quant_gather = ov::as_type_ptr<v8::Gather>(weight_scale_product->input_value(0).get_node_shared_ptr());
        auto scale_gather = ov::as_type_ptr<v8::Gather>(weight_scale_product->input_value(1).get_node_shared_ptr());
        auto packed_weights_gather = get_packed_mxfp4_gather(quant_gather->input_value(1));
        auto packed_scale_gather = get_packed_mxfp4_scale_gather(scale_gather->input_value(1));
        if (!packed_weights_gather || packed_weights_gather != packed_scale_gather ||
            packed_weights_gather->get_batch_dims() != 0 ||
            packed_weights_gather->input_value(1).get_element_type() != ov::element::i32 ||
            packed_weights_gather->input_value(1).get_partial_shape().rank() != 2 ||
            packed_weights_gather->input_value(0).get_partial_shape().rank() != 4 ||
            packed_weights_gather->get_output_partial_shape(0).rank() != 5) {
            return false;
        }

        std::shared_ptr<v0::Constant> weights;
        std::shared_ptr<v0::Constant> scales;
        if (!unpack_packed_mxfp4_weights(packed_weights_gather, weights, scales)) {
            return false;
        }

        const auto activations = expanded_activations->input_value(0);
        auto activation_transpose_order =
            v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto activation_transpose = std::make_shared<v1::Transpose>(activations, activation_transpose_order);
        auto empty_bias = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto empty_zero_points = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto gathered_matmul =
            std::make_shared<ov::op::internal::GatherMatmulCompressed>(activation_transpose,
                                                                       weights,
                                                                       packed_weights_gather->input_value(1),
                                                                       empty_bias,
                                                                       scales,
                                                                       empty_zero_points);
        auto output_transpose_order =
            v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto output_transpose = std::make_shared<v1::Transpose>(gathered_matmul, output_transpose_order);
        auto restored_rank_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
        auto restored_rank = std::make_shared<v0::Unsqueeze>(output_transpose, restored_rank_axis);

        ov::copy_runtime_info(matmul,
                              {activation_transpose_order,
                               activation_transpose,
                               empty_bias,
                               empty_zero_points,
                               gathered_matmul,
                               output_transpose_order,
                               output_transpose,
                               restored_rank_axis,
                               restored_rank});
        gathered_matmul->set_friendly_name(matmul->get_friendly_name());
        return ov::replace_output_update_name(matmul->output(0), restored_rank->output(0));
    };

    auto matcher = std::make_shared<pattern::Matcher>(matmul_pattern, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
