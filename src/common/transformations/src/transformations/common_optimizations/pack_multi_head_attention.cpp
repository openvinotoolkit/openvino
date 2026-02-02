// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/pack_multi_head_attention.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/concat_fusion.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass;

namespace {

static constexpr int64_t RANK = 4;  // Assuming 4D tensors [batch, num_heads, seq_len, hidden_dim]
static constexpr int64_t HEAD_AXIS = 1;

bool are_identical_nodes(const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node>& b) {
    if (!a || !b)
        return false;

    if (a->get_type_info() != b->get_type_info())
        return false;

    return true;
}

std::shared_ptr<ov::Node> get_packed_mha_concat(const std::shared_ptr<ov::Node>& root) {
    auto concat = ov::as_type_ptr<v0::Concat>(root);
    if (!concat || !concat->get_rt_info().count(PackMultiHeadAttentionRTInfo::get_type_info_static())) {
        return nullptr;
    }
    return concat;
};

/**
 * @brief Aligns the rank of a node by adding leading dimensions if necessary.
 *
 * This function ensures that the output node has at least the specified target rank.
 * If the current rank is less than the target rank, it adds leading dimensions
 * (of size 1) by applying an Unsqueeze operation.
 * If the current rank is already greater than or equal to the target rank,
 * the node is returned unchanged.
 *
 * @param output The output node whose rank needs to be aligned
 * @param target_rank The desired minimum rank for the output
 * @return A shared pointer to the node with aligned rank (either the original node
 *         or a new Unsqueeze node with added leading dimensions)
 */
std::shared_ptr<ov::Node> align_rank(const ov::Output<ov::Node>& output, int64_t target_rank) {
    auto pshape = output.get_partial_shape();
    if (pshape.rank().is_dynamic())
        return output.get_node_shared_ptr();

    int64_t cur_rank = pshape.rank().get_length();
    if (cur_rank >= target_rank)
        return output.get_node_shared_ptr();

    const int64_t preserved_dims = 2;  // Preserve last two dimensions (seq_len and hidden_dim)
    const int64_t insert_pos = std::max(cur_rank - preserved_dims, static_cast<int64_t>(0));
    std::vector<int64_t> axes(target_rank - cur_rank);
    std::iota(axes.begin(), axes.end(), insert_pos);

    auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    auto unsqueezed = ov::op::util::make_try_fold<v0::Unsqueeze>(output.get_node_shared_ptr(), axes_const);

    return unsqueezed;
}

/**
 * @brief Concatenates a vector of input tensors along a specified axis, aligning their ranks.
 *
 * This function first determines the maximum static rank among the input tensors (and the optional @p rank override).
 * It then aligns all input tensors to have at least the same rank (max_rank) before concatenation.
 * The concatenation is performed using the specified axis.
 *
 * @param inputs A vector of ov::Output objects representing the tensors to concatenate.
 * @param axis The axis along which to concatenate the tensors. Default is -1.
 * @param rank Optional minimum rank to align all inputs to before concatenation.
 * @return A shared pointer to the resulting Concat node.
 */
std::shared_ptr<ov::Node> concat_any(const ov::OutputVector& inputs, int64_t axis = -1, int64_t rank = 0) {
    int64_t max_rank = rank;

    for (const auto& t : inputs) {
        auto r = t.get_tensor().get_partial_shape().rank();
        if (r.is_static()) {
            max_rank = std::max(max_rank, r.get_length());
        } else {
            return nullptr;
        }
    }

    ov::OutputVector aligned_inputs;
    for (const auto& in : inputs) {
        aligned_inputs.push_back(align_rank(in, max_rank));
    }

    auto concat = ov::op::util::make_try_fold<v0::Concat>(aligned_inputs, axis);
    concat->get_rt_info().emplace(PackMultiHeadAttentionRTInfo::get_type_info_static(), PackMultiHeadAttentionRTInfo{});
    return concat;
}

/**
 * @brief Skips a node of specified type by returning its input node, or returns the node itself if it's not of the
 * specified type.
 *
 * @tparam T The type of node to skip (must be derived from ov::Node)
 * @param node The node to potentially skip
 * @return std::shared_ptr<ov::Node> If the node is of type T, returns its first input node; otherwise returns the node
 * itself
 *
 * This utility function attempts to cast the input node to type T. If successful, it bypasses the node
 * by returning its input (at index 0). If the cast fails, the original node is returned unchanged.
 * Useful for traversing node graphs while skipping specific operation types.
 */
template <typename T>
std::shared_ptr<ov::Node> skip_if_type(const std::shared_ptr<ov::Node>& node) {
    if (auto typed_node = ov::as_type_ptr<T>(node)) {
        return typed_node->input_value(0).get_node_shared_ptr();
    }
    return node;
};

// Helper function to extract scale node (supports both Divide and Multiply)
static std::shared_ptr<ov::Node> extract_scale_node(const std::shared_ptr<ov::Node>& input_node) {
    if (auto div = ov::as_type_ptr<v1::Divide>(input_node)) {
        return div;
    }
    if (auto mul = ov::as_type_ptr<v1::Multiply>(input_node)) {
        return mul;
    }
    return nullptr;
};

}  // namespace

bool PackMultiHeadAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PackMultiHeadAttention);

    ov::pass::Manager manager(get_pass_config(), "PackMultiHeadAttention");

    manager.register_pass<ov::pass::MergeUnrolledSDPA>();
    manager.register_pass<ov::pass::MergeKVCaches>();
    manager.register_pass<ov::pass::MergeUnrolledRoPE>();
    manager.register_pass<ov::pass::MergeLinearProjections>();
    manager.register_pass<ov::pass::MergeDQ>();
    manager.register_pass<ov::pass::ConcatFusion>();

    return manager.run_passes(model);
}

MergeUnrolledSDPA::MergeUnrolledSDPA() {
    MATCHER_SCOPE(MergeUnrolledSDPA);

    // Helper to create SDPA pattern
    auto create_sdpa_pattern = [&]() {
        auto q = pattern::any_input();
        auto k = pattern::any_input();
        auto v = pattern::any_input();
        auto k_transpose = pattern::optional<v1::Transpose>({k, pattern::any_input()});
        auto k_scale = pattern::optional<v1::Multiply>({k_transpose, pattern::any_input()});
        auto qk = pattern::wrap_type<v0::MatMul>({q, k_scale});
        auto qk_scale = pattern::optional<v1::Divide, v1::Multiply>({qk, pattern::any_input()});
        auto mask = pattern::wrap_type<v1::Add>({qk_scale, pattern::any_input()});
        auto softmax = pattern::wrap_type<v8::Softmax>({mask});
        auto qkv = pattern::wrap_type<v0::MatMul>({softmax, v});
        auto qkv_reshaped = pattern::optional<v1::Reshape>({qkv, pattern::any_input()});
        auto proj = pattern::any_input();
        auto proj_sf = pattern::optional<v1::Multiply>({proj, pattern::any_input()});
        auto matmul = pattern::wrap_type<v0::MatMul>({qkv_reshaped, proj_sf});
        auto matmul_reduced = pattern::optional<v1::ReduceSum>({matmul, pattern::any_input()});
        return matmul_reduced;
    };

    auto sdpa_lhs = create_sdpa_pattern();
    auto sdpa_rhs = create_sdpa_pattern();

    auto add = pattern::wrap_type<v1::Add>({sdpa_lhs, sdpa_rhs});

    auto m = std::make_shared<pattern::Matcher>(add, "MergeUnrolledSDPA");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto add_node = std::dynamic_pointer_cast<v1::Add>(matcher.get_match_root());
        if (!add_node)
            return false;

        // Extract MatMul nodes
        auto lhs = skip_if_type<v1::ReduceSum>(add_node->input_value(0).get_node_shared_ptr());
        auto rhs = skip_if_type<v1::ReduceSum>(add_node->input_value(1).get_node_shared_ptr());

        auto sdpa_proj1 = as_type_ptr<v0::MatMul>(lhs);
        auto sdpa_proj2 = as_type_ptr<v0::MatMul>(rhs);
        if (!are_identical_nodes(sdpa_proj1, sdpa_proj2))
            return false;

        // Get SDPA MatMul nodes (QKV)
        auto qk_v1 =
            as_type_ptr<v0::MatMul>(skip_if_type<v1::Reshape>(sdpa_proj1->input_value(0).get_node_shared_ptr()));
        auto qk_v2 =
            as_type_ptr<v0::MatMul>(skip_if_type<v1::Reshape>(sdpa_proj2->input_value(0).get_node_shared_ptr()));
        if (!are_identical_nodes(qk_v1, qk_v2))
            return false;

        // Extract Softmax nodes
        auto soft1 = as_type_ptr<v8::Softmax>(qk_v1->input_value(0).get_node_shared_ptr());
        auto soft2 = as_type_ptr<v8::Softmax>(qk_v2->input_value(0).get_node_shared_ptr());
        if (!are_identical_nodes(soft1, soft2))
            return false;

        // Extract bias Add nodes
        auto mask1 = as_type_ptr<v1::Add>(soft1->input_value(0).get_node_shared_ptr());
        auto mask2 = as_type_ptr<v1::Add>(soft2->input_value(0).get_node_shared_ptr());
        if (!are_identical_nodes(mask1, mask2))
            return false;

        // Extract optional scale nodes (Divide or Multiply)
        auto scale1 = extract_scale_node(mask1->input_value(0).get_node_shared_ptr());
        auto scale2 = extract_scale_node(mask2->input_value(0).get_node_shared_ptr());

        // Verify scales match if present
        if (scale1 && scale2) {
            auto c1 = as_type_ptr<v0::Constant>(scale1->input_value(1).get_node_shared_ptr());
            auto c2 = as_type_ptr<v0::Constant>(scale2->input_value(1).get_node_shared_ptr());
            if (!c1 || !c2 || c1->cast_vector<float>() != c2->cast_vector<float>())
                return false;
        } else if (scale1 || scale2) {
            // Only one scale present; cannot fuse
            return false;
        }

        auto qk1_node =
            scale1 ? scale1->input_value(0).get_node_shared_ptr() : mask1->input_value(0).get_node_shared_ptr();
        auto qk2_node =
            scale2 ? scale2->input_value(0).get_node_shared_ptr() : mask2->input_value(0).get_node_shared_ptr();

        auto qk1 = as_type_ptr<v0::MatMul>(qk1_node);
        auto qk2 = as_type_ptr<v0::MatMul>(qk2_node);
        if (!are_identical_nodes(qk1, qk2))
            return false;

        // Extract Q, K, V, sdpa_proj, mask from both SDPAs
        OutputVector q_inputs = {qk1->input_value(0), qk2->input_value(0)};
        OutputVector k_inputs = {qk1->input_value(1), qk2->input_value(1)};
        OutputVector v_inputs = {qk_v1->input_value(1), qk_v2->input_value(1)};
        OutputVector sdpa_proj_inputs = {sdpa_proj1->input_value(1), sdpa_proj2->input_value(1)};
        OutputVector mask_inputs = {mask1->input_value(1), mask2->input_value(1)};

        // Concatenate along head axis
        std::shared_ptr<ov::Node> Q = concat_any(q_inputs, HEAD_AXIS, RANK);
        std::shared_ptr<ov::Node> K = concat_any(k_inputs, HEAD_AXIS, RANK);
        std::shared_ptr<ov::Node> V = concat_any(v_inputs, HEAD_AXIS, RANK);
        std::shared_ptr<ov::Node> mask = (mask_inputs[0] == mask_inputs[1]) ? mask_inputs[0].get_node_shared_ptr()
                                                                            : concat_any(mask_inputs, HEAD_AXIS, RANK);
        std::shared_ptr<ov::Node> sdpa_proj = concat_any(sdpa_proj_inputs, HEAD_AXIS, RANK);

        // Check that concatenation succeeded; abort transformation on failure
        if (!Q || !K || !V || !mask || !sdpa_proj) {
            return false;
        }

        // Build fused SDPA
        auto qk_fused = qk1->copy_with_new_inputs({Q, K});
        copy_runtime_info({qk1, qk2}, qk_fused);

        std::shared_ptr<ov::Node> scores_scaled = qk_fused;
        if (scale1 && scale2) {
            scores_scaled = scale1->copy_with_new_inputs({qk_fused, scale1->input_value(1)});
            copy_runtime_info({scale1, scale2}, scores_scaled);
        }

        auto bias_fused = mask1->copy_with_new_inputs({scores_scaled, mask});
        copy_runtime_info({mask1, mask2}, bias_fused);

        auto softmax_fused = soft1->copy_with_new_inputs({bias_fused});
        copy_runtime_info({soft1, soft2}, softmax_fused);

        auto qkv_fused = qk_v1->copy_with_new_inputs({softmax_fused, V});
        copy_runtime_info({qk_v1, qk_v2}, qkv_fused);

        auto proj = sdpa_proj1->copy_with_new_inputs({qkv_fused, sdpa_proj});
        copy_runtime_info({sdpa_proj1, sdpa_proj2}, proj);

        auto reduce_axis = v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce = std::make_shared<v1::ReduceSum>(proj, reduce_axis, false);

        reduce->set_friendly_name(add_node->get_friendly_name() + "_reduced");
        copy_runtime_info(add_node, reduce);

        replace_node(add_node, reduce);

        return true;
    });
}

MergeUnrolledRoPE::MergeUnrolledRoPE() {
    MATCHER_SCOPE(MergeUnrolledRoPE);

    struct pattern_nodes {
        std::shared_ptr<Node> input;
        std::shared_ptr<Node> split;
        std::shared_ptr<Node> angle_r;
        std::shared_ptr<Node> concat;
        std::shared_ptr<Node> mul_l;
        std::shared_ptr<Node> mul_r;
        std::shared_ptr<Node> add;
        std::shared_ptr<Node> reshape;
        std::shared_ptr<Node> output;
    };

    // Helper to create RoPE pattern
    auto create_rope_pattern = [&]() {
        pattern_nodes nodes;
        auto input = pattern::any_input();
        nodes.input = pattern::wrap_type<v1::Reshape>({input, pattern::any_input()});
        nodes.split = pattern::wrap_type<v1::VariadicSplit>({nodes.input, pattern::any_input(), pattern::any_input()});
        nodes.split->set_output_size(2);
        nodes.angle_r = pattern::wrap_type<v0::Negative>({nodes.split});
        nodes.concat = pattern::wrap_type<v0::Concat>({nodes.angle_r, nodes.split});
        nodes.mul_r = pattern::wrap_type<v1::Multiply>({nodes.concat, pattern::any_input()});
        nodes.mul_l = pattern::wrap_type<v1::Multiply>({nodes.input, pattern::any_input()});
        nodes.add = pattern::wrap_type<v1::Add>({nodes.mul_l, nodes.mul_r});
        nodes.reshape = pattern::wrap_type<v1::Reshape, v1::Transpose>({nodes.add, pattern::any_input()});
        nodes.output = std::make_shared<pattern::op::Or>(OutputVector{nodes.reshape, nodes.add});
        return nodes;
    };

    auto rope_lhs = create_rope_pattern();
    auto rope_rhs = create_rope_pattern();

    auto concat = pattern::wrap_type<v0::Concat>({rope_lhs.output, rope_rhs.output});

    auto m = std::make_shared<pattern::Matcher>(concat, "MergeUnrolledRoPE");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = get_packed_mha_concat(matcher.get_match_root());
        if (!concat_node)
            return false;

        // Optional reshapes
        auto post_reshape_lhs = pm[rope_lhs.reshape].get_node_shared_ptr();
        auto post_reshape_rhs = pm[rope_rhs.reshape].get_node_shared_ptr();
        if (static_cast<bool>(post_reshape_lhs) != static_cast<bool>(post_reshape_rhs)) {
            return false;
        }

        auto add_lhs = as_type_ptr<v1::Add>(pm[rope_lhs.add].get_node_shared_ptr());
        auto add_rhs = as_type_ptr<v1::Add>(pm[rope_rhs.add].get_node_shared_ptr());
        if (!add_lhs || !add_rhs)
            return false;

        // extract Concat nodes from both RoPE patterns
        auto concat_lhs = as_type_ptr<v0::Concat>(pm[rope_lhs.concat].get_node_shared_ptr());
        auto concat_rhs = as_type_ptr<v0::Concat>(pm[rope_rhs.concat].get_node_shared_ptr());
        if (!concat_lhs || !concat_rhs)
            return false;

        // Extract optional angle nodes (Negative)
        auto angle_lhs = ov::as_type_ptr<v0::Negative>(concat_lhs->input_value(0).get_node_shared_ptr());
        auto angle_rhs = ov::as_type_ptr<v0::Negative>(concat_rhs->input_value(0).get_node_shared_ptr());
        if (!are_identical_nodes(angle_lhs, angle_rhs))
            return false;

        // extract Split nodes from both RoPE patterns
        auto split_lhs = as_type_ptr<v1::VariadicSplit>(pm[rope_lhs.split].get_node_shared_ptr());
        auto split_rhs = as_type_ptr<v1::VariadicSplit>(pm[rope_rhs.split].get_node_shared_ptr());
        if (!split_lhs || !split_rhs)
            return false;

        // Extract Multiply nodes from both RoPE patterns
        auto mul_down_1_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_r].get_node_shared_ptr());
        auto mul_down_1_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_r].get_node_shared_ptr());
        if (!mul_down_1_lhs || !mul_down_2_lhs || !mul_down_1_rhs || !mul_down_2_rhs)
            return false;

        // extract inputs
        auto reshape_lhs = pm[rope_lhs.input].get_node_shared_ptr();
        auto reshape_rhs = pm[rope_rhs.input].get_node_shared_ptr();

        // Concatenate along head axis
        auto input_fused =
            concat_any(ov::OutputVector{reshape_lhs->input_value(0), reshape_rhs->input_value(0)}, HEAD_AXIS, RANK);

        auto mul_down_1_lhs_input = mul_down_1_lhs->input_value(1);
        auto mul_down_2_lhs_input = mul_down_2_lhs->input_value(1);
        auto mul_down_1_rhs_input = mul_down_1_rhs->input_value(1);
        auto mul_down_2_rhs_input = mul_down_2_rhs->input_value(1);

        auto mul_down_input_l_fused =
            concat_any(ov::OutputVector{mul_down_1_lhs_input, mul_down_1_rhs_input}, HEAD_AXIS, RANK);

        auto mul_down_input_r_fused =
            concat_any(ov::OutputVector{mul_down_2_lhs_input, mul_down_2_rhs_input}, HEAD_AXIS, RANK);

        // Check that concatenation succeeded; abort transformation on failure
        if (!input_fused || !mul_down_input_r_fused || !mul_down_input_l_fused) {
            return false;
        }

        auto reshape_shape = v0::Constant::create(element::i64,
                                                  Shape{input_fused->get_output_shape(0).size()},
                                                  input_fused->get_output_shape(0));
        auto reshape_fused = reshape_lhs->copy_with_new_inputs({input_fused, reshape_shape});
        copy_runtime_info({reshape_lhs, reshape_rhs}, reshape_fused);

        auto split_fused =
            split_lhs->copy_with_new_inputs({reshape_fused, split_lhs->input_value(1), split_lhs->input_value(2)});
        copy_runtime_info({split_lhs, split_rhs}, split_fused);

        auto angle_fused = angle_lhs->copy_with_new_inputs({split_fused->output(1)});
        copy_runtime_info({angle_lhs, angle_rhs}, angle_fused);

        auto concat_fused = concat_lhs->copy_with_new_inputs({angle_fused, split_fused->output(0)});
        copy_runtime_info({concat_lhs, concat_rhs}, concat_fused);

        auto mul_down_l_fused = mul_down_1_lhs->copy_with_new_inputs({reshape_fused, mul_down_input_l_fused});
        copy_runtime_info({mul_down_1_lhs, mul_down_1_rhs}, mul_down_l_fused);

        auto mul_down_r_fused = mul_down_2_lhs->copy_with_new_inputs({concat_fused, mul_down_input_r_fused});
        copy_runtime_info({mul_down_2_lhs, mul_down_2_rhs}, mul_down_r_fused);

        auto add_fused = add_lhs->copy_with_new_inputs({mul_down_l_fused, mul_down_r_fused});
        copy_runtime_info({add_lhs, add_rhs}, add_fused);

        auto output_node = post_reshape_lhs
                               ? post_reshape_lhs->copy_with_new_inputs({add_fused, post_reshape_lhs->input_value(1)})
                               : add_fused;

        replace_node(concat_node, output_node);

        return true;
    });
}

MergeLinearProjections::MergeLinearProjections() {
    MATCHER_SCOPE(MergeLinearProjections);

    struct pattern_nodes {
        std::shared_ptr<Node> convert;
        std::shared_ptr<Node> subtract;
        std::shared_ptr<Node> multiply;
        std::shared_ptr<Node> matmul;
        std::shared_ptr<Node> add;
        std::shared_ptr<Node> output;
    };

    // Helper to create pattern
    auto create_matmul_bias_pattern = [&](const std::shared_ptr<Node>& input) {
        pattern_nodes nodes;
        nodes.convert = pattern::optional<v0::Convert>({pattern::any_input()});
        nodes.subtract = pattern::optional<v1::Subtract>({nodes.convert, pattern::any_input()});
        nodes.multiply = pattern::optional<v1::Multiply>({nodes.subtract, pattern::any_input()});
        nodes.matmul = pattern::wrap_type<v0::MatMul>({input, nodes.multiply});
        nodes.add = pattern::wrap_type<v1::Add>({nodes.matmul, pattern::any_input()});
        auto reshape = pattern::wrap_type<v1::Reshape, v0::Unsqueeze>({nodes.add, pattern::any_input()});
        nodes.output = std::make_shared<pattern::op::Or>(OutputVector{reshape, nodes.add});
        return nodes;
    };

    auto input = pattern::any_input();
    auto input_unsqueezed = pattern::optional<v0::Unsqueeze>({input, pattern::any_input()});

    auto mm_bias_lhs = create_matmul_bias_pattern(input_unsqueezed);
    auto mm_bias_rhs = create_matmul_bias_pattern(input_unsqueezed);

    auto concat = pattern::wrap_type<v0::Concat>({mm_bias_lhs.output, mm_bias_rhs.output});
    auto m = std::make_shared<pattern::Matcher>(concat, "MergeLinearProjections");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = get_packed_mha_concat(matcher.get_match_root());
        if (!concat_node) {
            return false;
        }

        auto mm_lhs = as_type_ptr<v0::MatMul>(pm[mm_bias_lhs.matmul].get_node_shared_ptr());
        auto mm_rhs = as_type_ptr<v0::MatMul>(pm[mm_bias_rhs.matmul].get_node_shared_ptr());
        if (!mm_lhs || !mm_rhs) {
            return false;
        }

        auto add_lhs = as_type_ptr<v1::Add>(pm[mm_bias_lhs.add].get_node_shared_ptr());
        auto add_rhs = as_type_ptr<v1::Add>(pm[mm_bias_rhs.add].get_node_shared_ptr());
        if (!add_lhs || !add_rhs)
            return false;

        // Concatenate along head axis
        std::shared_ptr<ov::Node> input_fused = nullptr;
        auto convert_lhs = ov::as_type_ptr<v0::Convert>(pm[mm_bias_lhs.convert].get_node_shared_ptr());
        auto convert_rhs = ov::as_type_ptr<v0::Convert>(pm[mm_bias_rhs.convert].get_node_shared_ptr());
        if (convert_lhs && convert_rhs) {
            auto convert_const_fused =
                concat_any(OutputVector{convert_lhs->input_value(0), convert_rhs->input_value(0)}, HEAD_AXIS, RANK);
            if (!convert_const_fused) {
                return false;
            }
            input_fused = convert_lhs->copy_with_new_inputs({convert_const_fused});
            copy_runtime_info({convert_lhs, convert_rhs}, input_fused);
        }

        auto subtract_lhs = ov::as_type_ptr<v1::Subtract>(pm[mm_bias_lhs.subtract].get_node_shared_ptr());
        auto subtract_rhs = ov::as_type_ptr<v1::Subtract>(pm[mm_bias_rhs.subtract].get_node_shared_ptr());
        if (subtract_lhs && subtract_rhs) {
            auto subtract_const_fused =
                concat_any(OutputVector{subtract_lhs->input_value(0), subtract_rhs->input_value(0)}, HEAD_AXIS, RANK);
            if (!subtract_const_fused) {
                return false;
            }
            input_fused = subtract_lhs->copy_with_new_inputs({subtract_const_fused, subtract_lhs->input_value(1)});
            copy_runtime_info({subtract_lhs, subtract_rhs}, input_fused);
        }

        auto scale_lhs = ov::as_type_ptr<v1::Multiply>(pm[mm_bias_lhs.multiply].get_node_shared_ptr());
        auto scale_rhs = ov::as_type_ptr<v1::Multiply>(pm[mm_bias_rhs.multiply].get_node_shared_ptr());
        if (scale_lhs && scale_rhs) {
            if (!input_fused) {
                OutputVector multiply_inputs = {scale_lhs->input_value(0), scale_rhs->input_value(0)};
                input_fused = concat_any(multiply_inputs, HEAD_AXIS, RANK);
                if (!input_fused) {
                    return false;
                }
            }
            input_fused = scale_lhs->copy_with_new_inputs({input_fused, scale_lhs->input_value(1)});
            copy_runtime_info({scale_lhs, scale_rhs}, input_fused);
        }

        if (!input_fused) {
            input_fused = concat_any(OutputVector{mm_lhs->input_value(1), mm_rhs->input_value(1)}, HEAD_AXIS, RANK);
            if (!input_fused) {
                return false;
            }
        }

        auto mm_fused = mm_lhs->copy_with_new_inputs({align_rank(mm_lhs->input_value(0), RANK), input_fused});
        copy_runtime_info({mm_lhs, mm_rhs}, mm_fused);

        auto bias_fused = concat_any(OutputVector{add_lhs->input_value(1), add_rhs->input_value(1)}, HEAD_AXIS, RANK);
        if (!bias_fused) {
            return false;
        }
        auto add_fused = add_lhs->copy_with_new_inputs({mm_fused, bias_fused});
        copy_runtime_info({add_lhs, add_rhs}, add_fused);

        replace_node(concat_node, add_fused);

        return true;
    });
}

MergeDQ::MergeDQ() {
    MATCHER_SCOPE(MergeDQ);

    struct pattern_nodes {
        std::shared_ptr<Node> convert_0;
        std::shared_ptr<Node> convert_1;
        std::shared_ptr<Node> subtract;
        std::shared_ptr<Node> multiply;
        std::shared_ptr<Node> output;
    };

    // Helper to create pattern
    auto create_dq_pattern = [&]() {
        pattern_nodes nodes;
        auto input_0 = pattern::wrap_type<v0::Constant>();
        auto input_1 = pattern::wrap_type<v0::Constant>();
        nodes.convert_0 = pattern::wrap_type<v0::Convert>({input_0});
        nodes.convert_1 = pattern::wrap_type<v0::Convert>({input_1});
        nodes.subtract = pattern::wrap_type<v1::Subtract>({nodes.convert_0, nodes.convert_1});
        nodes.multiply = pattern::wrap_type<v1::Multiply>({nodes.subtract, pattern::any_input()});
        nodes.output = pattern::optional<v1::Reshape, v0::Unsqueeze>({nodes.multiply, pattern::any_input()});
        return nodes;
    };

    auto dq_lhs = create_dq_pattern();
    auto dq_rhs = create_dq_pattern();

    auto concat = pattern::wrap_type<v0::Concat>({dq_lhs.output, dq_rhs.output});
    auto m = std::make_shared<pattern::Matcher>(concat, "MergeDQ");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = get_packed_mha_concat(matcher.get_match_root());
        if (!concat_node) {
            return false;
        }

        auto mul_lhs = as_type_ptr<v1::Multiply>(pm[dq_lhs.multiply].get_node_shared_ptr());
        auto mul_rhs = as_type_ptr<v1::Multiply>(pm[dq_rhs.multiply].get_node_shared_ptr());
        if (!mul_lhs || !mul_rhs) {
            return false;
        }

        auto sub_lhs = as_type_ptr<v1::Subtract>(pm[dq_lhs.subtract].get_node_shared_ptr());
        auto sub_rhs = as_type_ptr<v1::Subtract>(pm[dq_rhs.subtract].get_node_shared_ptr());
        if (!sub_lhs || !sub_rhs)
            return false;

        auto convert_0_lhs = as_type_ptr<v0::Convert>(pm[dq_lhs.convert_0].get_node_shared_ptr());
        auto convert_0_rhs = as_type_ptr<v0::Convert>(pm[dq_rhs.convert_0].get_node_shared_ptr());
        auto convert_1_lhs = as_type_ptr<v0::Convert>(pm[dq_lhs.convert_1].get_node_shared_ptr());
        auto convert_1_rhs = as_type_ptr<v0::Convert>(pm[dq_rhs.convert_1].get_node_shared_ptr());
        if (!convert_0_lhs || !convert_0_rhs || !convert_1_lhs || !convert_1_rhs)
            return false;

        // Concatenate along head axis
        auto convert_0_const_fused =
            concat_any(OutputVector{convert_0_lhs->input_value(0), convert_0_rhs->input_value(0)}, HEAD_AXIS, RANK);
        auto convert_1_const_fused =
            concat_any(OutputVector{convert_1_lhs->input_value(0), convert_1_rhs->input_value(0)}, HEAD_AXIS, RANK);

        // Check that concatenation succeeded; abort transformation on failure
        if (!convert_0_const_fused || !convert_1_const_fused) {
            return false;
        }

        auto convert_0_fused = convert_0_lhs->copy_with_new_inputs({convert_0_const_fused});
        copy_runtime_info({convert_0_lhs, convert_0_rhs}, convert_0_fused);

        auto convert_1_fused = convert_1_lhs->copy_with_new_inputs({convert_1_const_fused});
        copy_runtime_info({convert_1_lhs, convert_1_rhs}, convert_1_fused);

        auto sub_fused = sub_lhs->copy_with_new_inputs({convert_0_fused, convert_1_fused});
        copy_runtime_info({sub_lhs, sub_rhs}, sub_fused);

        auto scale_fused = concat_any(OutputVector{mul_lhs->input_value(1), mul_rhs->input_value(1)}, HEAD_AXIS, RANK);
        if (!scale_fused) {
            return false;
        }

        auto mul_fused = mul_lhs->copy_with_new_inputs(OutputVector{sub_fused, scale_fused});
        copy_runtime_info({mul_lhs, mul_rhs}, mul_fused);

        replace_node(concat_node, mul_fused);

        return true;
    });
}

MergeKVCaches::MergeKVCaches() {
    MATCHER_SCOPE(MergeKVCaches);

    // Helper to create cache pattern
    auto create_cache_pattern = [&]() {
        auto cache_input = pattern::wrap_type<v0::Parameter>({});
        auto input = pattern::any_input();
        auto concat = pattern::wrap_type<v0::Concat>({cache_input, input});
        return concat;
    };

    auto cache_lhs = create_cache_pattern();
    auto cache_rhs = create_cache_pattern();

    auto concat = pattern::wrap_type<v0::Concat>({cache_lhs, cache_rhs});

    auto m = std::make_shared<pattern::Matcher>(concat, "MergeKVCaches");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = get_packed_mha_concat(matcher.get_match_root());
        if (!concat_node) {
            return false;
        }

        auto concat_cache_lhs = ov::as_type_ptr<v0::Concat>(pm[cache_lhs].get_node_shared_ptr());
        auto concat_cache_rhs = ov::as_type_ptr<v0::Concat>(pm[cache_rhs].get_node_shared_ptr());
        if (!concat_cache_lhs || !concat_cache_rhs)
            return false;

        // Concatenate along head axis
        auto concat_merged_input1 =
            concat_any(OutputVector{concat_cache_lhs->input_value(0), concat_cache_rhs->input_value(0)},
                       HEAD_AXIS,
                       RANK);
        auto concat_merged_input2 =
            concat_any(OutputVector{concat_cache_lhs->input_value(1), concat_cache_rhs->input_value(1)},
                       HEAD_AXIS,
                       RANK);

        // Check that concatenation succeeded
        if (!concat_merged_input1 || !concat_merged_input2) {
            return false;
        }
        auto concat_merged = concat_cache_lhs->copy_with_new_inputs({concat_merged_input1, concat_merged_input2});
        copy_runtime_info({concat_cache_lhs, concat_cache_rhs}, concat_merged);

        // splitting cache output
        const auto& pshape_lhs = concat_node->input_value(0).get_partial_shape();
        const auto& pshape_rhs = concat_node->input_value(1).get_partial_shape();

        // Ensure ranks and the head dimension are static before accessing lengths
        if (pshape_lhs.rank().is_dynamic() || pshape_rhs.rank().is_dynamic())
            return false;
        if (HEAD_AXIS >= pshape_lhs.size() || HEAD_AXIS >= pshape_rhs.size())
            return false;
        if (pshape_lhs[HEAD_AXIS].is_dynamic() || pshape_rhs[HEAD_AXIS].is_dynamic())
            return false;

        int64_t len_lhs = pshape_lhs[HEAD_AXIS].get_length();
        int64_t len_rhs = pshape_rhs[HEAD_AXIS].get_length();

        auto axis = v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>({HEAD_AXIS}));
        auto sizes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {len_lhs, len_rhs});
        auto split_out = std::make_shared<v1::VariadicSplit>(concat_merged, axis, sizes);

        replace_node(concat_node, concat_merged);

        // replace concat consumers
        auto replace_consumers = [&split_out](const std::shared_ptr<Node>& node, size_t output_idx) {
            for (auto& target_input : node->get_output_target_inputs(0)) {
                target_input.replace_source_output(split_out->output(output_idx));
            }
        };

        replace_consumers(concat_cache_lhs, 0);
        replace_consumers(concat_cache_rhs, 1);

        return true;
    });
}