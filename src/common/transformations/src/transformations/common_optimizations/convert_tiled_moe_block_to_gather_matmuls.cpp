// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

#include <initializer_list>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
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
#include "ov_ops/gather_matmul.hpp"

namespace {
using namespace ov::pass;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

// Note: intermediate nodes remain unchanged,
// but we need to explicitly call shape inference for them to keep shapes consistency
void validate_nodes(const pattern::PatternValueMap& map, const std::initializer_list<std::shared_ptr<ov::Node>> nodes) {
    for (const auto& node : nodes) {
        map.at(node).get_node_shared_ptr()->validate_and_infer_types();
    }
};

std::shared_ptr<ov::op::v0::Unsqueeze> introduce_n_experts_dim(const ov::Output<ov::Node>& data) {
    auto zero_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(data, zero_const);
    ov::copy_runtime_info(data.get_node_shared_ptr(), {unsqueeze, zero_const});
    return unsqueeze;
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

MOE2GEMMPatternNodes build_2gemm_pattern() {
    MOE2GEMMPatternNodes p;

    p.experts_input = pattern::wrap_type<v1::Reshape>({pattern::any_input(), pattern::any_input()});
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()});
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()});
    p.gate_up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.gate_up_bias = pattern::wrap_const();
    p.gate_up_add = pattern::wrap_type<v1::Add>({p.gate_up_matmul, p.gate_up_bias}, pattern::consumers_count(2));

    // Branch 1: Slice_1 -> Clamp -> Add_1
    p.slice1 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    p.clamp = pattern::wrap_type<v0::Clamp>({p.slice1});
    p.add1 = pattern::wrap_type<v1::Add>({p.clamp, pattern::wrap_const()});

    // Branch 2: Slice_2 -> Minimum_1 -> Swish
    p.slice2 = pattern::wrap_type<v8::Slice>(
        {p.gate_up_add, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    p.minimum1 = pattern::wrap_type<v1::Minimum>({p.slice2, pattern::wrap_const()});
    p.swish_beta = pattern::wrap_const();
    p.swish = pattern::wrap_type<v4::Swish>({p.minimum1, p.swish_beta});

    // Join: Multiply_2
    p.multiply2 = pattern::wrap_type<v1::Multiply>({p.add1, p.swish});

    // Down projection
    p.down_proj_matmul = pattern::wrap_type<v0::MatMul>(
        {p.multiply2, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.down_proj_bias = pattern::wrap_const();
    p.down_proj_add = pattern::wrap_type<v1::Add>({p.down_proj_matmul, p.down_proj_bias});
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape = pattern::wrap_type<v1::Reshape>({p.down_proj_add, p.end_reshape_target_shape});

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

MOE3GEMMPatternNodes build_3gemm_pattern() {
    MOE3GEMMPatternNodes p;

    p.experts_input = pattern::wrap_type<v1::Reshape>({pattern::any_input(), pattern::any_input()});
    p.tile = pattern::wrap_type<v0::Tile>({p.experts_input, pattern::any_input()});
    p.after_tile_reshape = pattern::wrap_type<v1::Reshape>({p.tile, pattern::any_input()});

    // First GEMM (activation gate)
    p.gate_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.swish = pattern::wrap_type<v4::Swish>({p.gate_matmul});
    // Second GEMM (up_projection)
    p.up_matmul = pattern::wrap_type<v0::MatMul>(
        {p.after_tile_reshape, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    // Join: Multiply (SwiGLU)
    p.swiglu = pattern::wrap_type<v1::Multiply>({p.swish, p.up_matmul});

    // Third GEMM (down_projection)
    p.down_matmul = pattern::wrap_type<v0::MatMul>(
        {p.swiglu, pattern::any_input()},
        pattern::consumers_count(1) && pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));
    p.end_reshape_target_shape = pattern::any_input();
    p.end_reshape = pattern::wrap_type<v1::Reshape>({p.down_matmul, p.end_reshape_target_shape});

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

ConvertTiledMoeBlockTo2GatherMatmuls::ConvertTiledMoeBlockTo2GatherMatmuls() {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo2GatherMatmuls);

    auto p = build_2gemm_pattern();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
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
        if (pm.find(p.reduceSum_squeeze) != pm.end()) {
            const auto reduce_node = pm.at(p.reduceSum_keepDims).get_node_shared_ptr();
            const auto squeeze_node = pm.at(p.reduceSum_squeeze).get_node_shared_ptr();
            const auto new_reduce_node =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_node);
            new_reduce_sum =
                squeeze_node->clone_with_new_inputs({new_reduce_node->output(0), squeeze_node->input_value(1)});
            ov::copy_runtime_info(squeeze_node, new_reduce_sum);
            new_reduce_sum->set_friendly_name(squeeze_node->get_friendly_name());
        } else {
            const auto reduce_node = pm.at(p.reduceSum_noKeepDims).get_node_shared_ptr();
            new_reduce_sum =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_sum);
            new_reduce_sum->set_friendly_name(reduce_node->get_friendly_name());
        }

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

ConvertTiledMoeBlockTo3GatherMatmuls::ConvertTiledMoeBlockTo3GatherMatmuls() {
    MATCHER_SCOPE(ConvertTiledMoeBlockTo3GatherMatmuls);

    auto p = build_3gemm_pattern();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
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
        if (pm.find(p.reduceSum_squeeze) != pm.end()) {
            const auto reduce_node = pm.at(p.reduceSum_keepDims).get_node_shared_ptr();
            const auto squeeze_node = pm.at(p.reduceSum_squeeze).get_node_shared_ptr();
            const auto new_reduce_node =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_node);
            new_reduce_sum =
                squeeze_node->clone_with_new_inputs({new_reduce_node->output(0), squeeze_node->input_value(1)});
            ov::copy_runtime_info(squeeze_node, new_reduce_sum);
            new_reduce_sum->set_friendly_name(squeeze_node->get_friendly_name());
        } else {
            const auto reduce_node = pm.at(p.reduceSum_noKeepDims).get_node_shared_ptr();
            new_reduce_sum =
                reduce_node->clone_with_new_inputs({new_final_mul->output(0), reduce_node->input_value(1)});
            ov::copy_runtime_info(reduce_node, new_reduce_sum);
            new_reduce_sum->set_friendly_name(reduce_node->get_friendly_name());
        }

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

}  // namespace ov::pass
