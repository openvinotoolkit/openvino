// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/pack_GQA.hpp"

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
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations/common_optimizations/concat_fusion.hpp"
#include "transformations/utils/block_collection.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass;

namespace {

bool compare_nodes(const std::shared_ptr<ov::Node>& a, const std::shared_ptr<ov::Node>& b) {
    if (!a || !b)
        return false;

    if (a->get_type_info() != b->get_type_info())
        return false;

    // if (a->get_output_size() != b->get_output_size())
    //     return false;

    // for (size_t i = 0; i < a->get_output_size(); ++i) {
    //     if (a->get_output_shape(i) != b->get_output_shape(i))
    //         return false;
    // }

    return true;
}

/**
 * @brief Normalizes the rank of a node by adding leading dimensions if necessary.
 *
 * This function ensures that the output node has at least the specified target rank.
 * If the current rank is less than the target rank, it adds leading dimensions
 * (of size 1) by applying an Unsqueeze operation at the front of the shape.
 * If the current rank is already greater than or equal to the target rank,
 * the node is returned unchanged.
 *
 * @param output The output node whose rank needs to be normalized
 * @param target_rank The desired minimum rank for the output
 * @return A shared pointer to the node with normalized rank (either the original node
 *         or a new Unsqueeze node with added leading dimensions)
 */
std::shared_ptr<ov::Node> normalize_rank(const ov::Output<ov::Node>& output, int64_t target_rank) {
    auto pshape = output.get_partial_shape();
    if (pshape.rank().is_dynamic())
        return output.get_node_shared_ptr();

    int64_t cur_rank = pshape.rank().get_length();
    if (cur_rank >= target_rank)
        return output.get_node_shared_ptr();

    std::vector<int64_t> axes;
    axes.reserve(target_rank - cur_rank);
    for (int64_t i = 0; i < target_rank - cur_rank; ++i)
        axes.push_back(i);

    auto axes_const = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    auto unsqueezed = util::make_try_fold<v0::Unsqueeze>(output.get_node_shared_ptr(), axes_const);

    return unsqueezed;
}

/**
 * @brief Concatenates a vector of input tensors along a specified axis, normalizing their ranks.
 *
 * This function first determines the maximum static rank among the input tensors.
 * It then normalizes all input tensors to have the same rank (max_rank + 1) before concatenation.
 * The concatenation is performed using the specified axis.
 *
 * @param inputs A vector of ov::Output objects representing the tensors to concatenate.
 * @param axis The axis along which to concatenate the tensors. Default is 0.
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

    ov::OutputVector normalized;
    for (const auto& t : inputs) {
        normalized.push_back(normalize_rank(t, max_rank));
    }

    auto concat = util::make_try_fold<v0::Concat>(normalized, axis);
    concat->get_rt_info()["packed_GQA"] = true;

    return concat;
}

}  // namespace

bool PackGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PackGQA);
    ov::pass::Manager manager(get_pass_config(), "PackGQA");

    manager.register_pass<ov::pass::Serialize>("PackGQA_before.xml", "PackGQA_before.bin");
    manager.register_pass<ov::pass::MergeTwoUnrolledSDPAAdd>();
    manager.register_pass<ov::pass::MergeKVCaches>();
    manager.register_pass<ov::pass::MergeTwoUnrolledRoPEConcat>();
    manager.register_pass<ov::pass::MergeMatMulBiasConcat>();
    manager.register_pass<ov::pass::MergeDQConcat>();
    manager.register_pass<ov::pass::ConcatFusion>();
    // manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::Serialize>("PackGQA_after.xml", "PackGQA_after.bin");

    return manager.run_passes(model);
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
std::shared_ptr<ov::Node> skip_node(const std::shared_ptr<ov::Node>& node) {
    if (auto typed_node = ov::as_type_ptr<T>(node)) {
        return typed_node->input_value(0).get_node_shared_ptr();
    }
    return node;
};

// Helper function to extract scale node (supports both Divide and Multiply)
static std::shared_ptr<ov::Node> get_scale(const std::shared_ptr<ov::Node>& bias_node) {
    auto input_node = bias_node->input_value(0).get_node_shared_ptr();
    if (auto div = ov::as_type_ptr<v1::Divide>(input_node)) {
        return div;
    }
    if (auto mul = ov::as_type_ptr<v1::Multiply>(input_node)) {
        return mul;
    }
    return nullptr;
};

/**
 * @brief Merges two unrolled Scaled Dot-Product Attention (SDPA) patterns connected by an Add operation.
 *
 * This transformation identifies and fuses two separate SDPA subgraphs that are added together,
 * which commonly occurs in multi-head attention mechanisms with split heads. The pattern matches:
 *
 * Pattern for each SDPA branch:
 * - Q @ K (optional K scale via Multiply)
 * - Optional scale (Divide/Multiply)
 * - Add bias
 * - Softmax
 * - @ V
 * - Optional Reshape
 * - @ Projection
 * - Optional ReduceSum
 * - Final Add combining both branches
 *
 * The transformation concatenates the Q, K, V, Bias, and Projection tensors along the head axis (axis=1)
 * and rebuilds a single fused SDPA operation followed by a ReduceSum, reducing memory footprint and
 * computational overhead.
 *
 * @note This optimization is beneficial for Grouped Query Attention (GQA) and Multi-Head Attention (MHA)
 *       patterns where heads are processed separately then combined.
 * @note The transformation verifies that scale factors match between both branches if present.
 * @note Bias tensors are only concatenated if they differ; otherwise the same bias is reused.
 */
MergeTwoUnrolledSDPAAdd::MergeTwoUnrolledSDPAAdd() {
    MATCHER_SCOPE(MergeTwoUnrolledSDPAAdd);

    // Helper to create SDPA pattern
    auto create_sdpa_pattern = [&]() {
        auto q = pattern::any_input();
        auto k = pattern::any_input();
        auto v = pattern::any_input();
        auto k_scale = pattern::optional<v1::Multiply>({k, pattern::any_input()});
        auto qk = pattern::wrap_type<v0::MatMul>({q, k_scale});
        auto qk_scale = pattern::optional<v1::Divide, v1::Multiply>({qk, pattern::any_input()});
        auto bias_add = pattern::wrap_type<v1::Add>({qk_scale, pattern::any_input()});
        auto softmax = pattern::wrap_type<v8::Softmax>({bias_add});
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

    auto m = std::make_shared<pattern::Matcher>(add, "MergeTwoUnrolledSDPAAdd");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto add_node = std::dynamic_pointer_cast<v1::Add>(matcher.get_match_root());
        if (!add_node)
            return false;

        // Extract MatMul nodes
        auto lhs = skip_node<v1::ReduceSum>(add_node->input_value(0).get_node_shared_ptr());
        auto rhs = skip_node<v1::ReduceSum>(add_node->input_value(1).get_node_shared_ptr());

        auto mm1 = as_type_ptr<v0::MatMul>(lhs);
        auto mm2 = as_type_ptr<v0::MatMul>(rhs);
        if (!compare_nodes(mm1, mm2))
            return false;

        // Get SDPA MatMul nodes (QKV)
        auto sdpa_mm1 = as_type_ptr<v0::MatMul>(skip_node<v1::Reshape>(mm1->input_value(0).get_node_shared_ptr()));
        auto sdpa_mm2 = as_type_ptr<v0::MatMul>(skip_node<v1::Reshape>(mm2->input_value(0).get_node_shared_ptr()));
        if (!compare_nodes(sdpa_mm1, sdpa_mm2))
            return false;

        // Extract Softmax nodes
        auto soft1 = as_type_ptr<v8::Softmax>(sdpa_mm1->input_value(0).get_node_shared_ptr());
        auto soft2 = as_type_ptr<v8::Softmax>(sdpa_mm2->input_value(0).get_node_shared_ptr());
        if (!compare_nodes(soft1, soft2))
            return false;

        // Extract bias Add nodes
        auto bias1 = as_type_ptr<v1::Add>(soft1->input_value(0).get_node_shared_ptr());
        auto bias2 = as_type_ptr<v1::Add>(soft2->input_value(0).get_node_shared_ptr());
        if (!compare_nodes(bias1, bias2))
            return false;

        // Extract optional scale nodes (Divide or Multiply)
        auto sf1 = get_scale(bias1);
        auto sf2 = get_scale(bias2);

        // Verify scales match if present
        if (sf1 && sf2) {
            auto c1 = as_type_ptr<v0::Constant>(sf1->input_value(1).get_node_shared_ptr());
            auto c2 = as_type_ptr<v0::Constant>(sf2->input_value(1).get_node_shared_ptr());
            if (!c1 || !c2 || c1->cast_vector<float>() != c2->cast_vector<float>())
                return false;
        }

        auto qk1_node = sf1 ? sf1->input_value(0).get_node_shared_ptr() : bias1->input_value(0).get_node_shared_ptr();
        auto qk2_node = sf2 ? sf2->input_value(0).get_node_shared_ptr() : bias2->input_value(0).get_node_shared_ptr();

        auto qk1 = as_type_ptr<v0::MatMul>(qk1_node);
        auto qk2 = as_type_ptr<v0::MatMul>(qk2_node);
        if (!compare_nodes(qk1, qk2))
            return false;

        // Extract Q, K, V, P, B from both SDPAs
        Output<Node> Q1 = qk1->input_value(0), K1 = qk1->input_value(1);
        Output<Node> V1 = sdpa_mm1->input_value(1), P1 = mm1->input_value(1), B1 = bias1->input_value(1);

        Output<Node> Q2 = qk2->input_value(0), K2 = qk2->input_value(1);
        Output<Node> V2 = sdpa_mm2->input_value(1), P2 = mm2->input_value(1), B2 = bias2->input_value(1);

        // Concatenate along head axis (1)
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]
        auto Q = concat_any(OutputVector{Q1, Q2}, head_axis, rank);
        auto K = concat_any(OutputVector{K1, K2}, head_axis, rank);
        auto V = concat_any(OutputVector{V1, V2}, head_axis, rank);
        auto B = (B1 == B2) ? B1 : concat_any(OutputVector{B1, B2}, head_axis, rank);
        auto P = concat_any(OutputVector{P1, P2}, head_axis, rank);

        // Build fused SDPA
        auto qk_fused = qk1->copy_with_new_inputs({Q, K});
        std::shared_ptr<ov::Node> scores_scaled = qk_fused;
        if (sf1 && sf2) {
            scores_scaled = sf1->copy_with_new_inputs({qk_fused, sf1->input_value(1)});
        }

        auto bias_fused = bias1->copy_with_new_inputs({scores_scaled, B});
        auto softmax_fused = soft1->copy_with_new_inputs({bias_fused});
        auto qkv_fused = sdpa_mm1->copy_with_new_inputs({softmax_fused, V});
        auto proj = mm1->copy_with_new_inputs({qkv_fused, P});

        auto reduce_axis = v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce = std::make_shared<v1::ReduceSum>(proj, reduce_axis, false);

        reduce->set_friendly_name(add_node->get_friendly_name() + "_reduced");
        copy_runtime_info({add_node, mm1, mm2, soft1, soft2}, reduce);

        replace_node(add_node, reduce);

        return true;
    });
}

MergeTwoUnrolledRoPEConcat::MergeTwoUnrolledRoPEConcat() {
    MATCHER_SCOPE(MergeTwoUnrolledRoPEConcat);

    struct pattern_nodes {
        std::shared_ptr<Node> input;
        std::shared_ptr<Node> scale;
        std::shared_ptr<Node> mul_l;
        std::shared_ptr<Node> mul_r;
        std::shared_ptr<Node> add;
        std::shared_ptr<Node> output;
    };

    // Helper to create RoPE pattern
    auto create_rope_pattern = [&]() {
        pattern_nodes nodes;
        auto input = pattern::any_input();
        nodes.input = pattern::wrap_type<v1::Reshape>({input, pattern::any_input()});
        auto var_split =
            pattern::wrap_type<v1::VariadicSplit>({nodes.input, pattern::any_input(), pattern::any_input()});
        var_split->set_output_size(2);

        nodes.scale = pattern::wrap_type<v0::Negative>({var_split->output(1)});
        auto concat = pattern::wrap_type<v0::Concat>({nodes.scale, var_split->output(0)});
        nodes.mul_l = pattern::wrap_type<v1::Multiply>({concat, pattern::any_input()});
        nodes.mul_r = pattern::wrap_type<v1::Multiply>({nodes.input, pattern::any_input()});
        nodes.add = pattern::wrap_type<v1::Add>({nodes.mul_r, nodes.mul_l});
        auto reshape = pattern::wrap_type<v1::Reshape>({nodes.add, pattern::any_input()});
        nodes.output = std::make_shared<pattern::op::Or>(OutputVector{reshape, nodes.add});

        return nodes;
    };

    auto rope_lhs = create_rope_pattern();
    auto rope_rhs = create_rope_pattern();

    auto concat = pattern::wrap_type<v0::Concat>({rope_lhs.output, rope_rhs.output});

    auto m = std::make_shared<pattern::Matcher>(concat, "MergeTwoUnrolledRoPEConcat");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = std::dynamic_pointer_cast<v0::Concat>(matcher.get_match_root());
        if (!concat_node || !concat_node->get_rt_info().count("packed_GQA"))
            return false;

        auto add_lhs = as_type_ptr<v1::Add>(pm[rope_lhs.add].get_node_shared_ptr());
        auto add_rhs = as_type_ptr<v1::Add>(pm[rope_rhs.add].get_node_shared_ptr());
        if (!add_lhs || !add_rhs)
            return false;

        // Extract Multiply nodes from both RoPE patterns
        auto mul_down_1_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_r].get_node_shared_ptr());
        auto mul_down_1_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_r].get_node_shared_ptr());
        if (!mul_down_1_lhs || !mul_down_2_lhs || !mul_down_1_rhs || !mul_down_2_rhs)
            return false;

        // Extract optional scale nodes (Negative)
        auto sf_lhs = as_type_ptr<v0::Negative>(pm[rope_lhs.scale].get_node_shared_ptr());
        auto sf_rhs = as_type_ptr<v0::Negative>(pm[rope_rhs.scale].get_node_shared_ptr());

        // extract inputs
        auto reshape_lhs = pm[rope_lhs.input].get_node_shared_ptr();
        auto reshape_rhs = pm[rope_rhs.input].get_node_shared_ptr();

        // Concatenate along head axis (1)
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]

        auto input_fused =
            concat_any(ov::OutputVector{reshape_lhs->input_value(0), reshape_rhs->input_value(0)}, head_axis, rank);

        auto mul_down_1_lhs_input = mul_down_1_lhs->input_value(1);
        auto mul_down_2_lhs_input = mul_down_2_lhs->input_value(1);
        auto mul_down_1_rhs_input = mul_down_1_rhs->input_value(1);
        auto mul_down_2_rhs_input = mul_down_2_rhs->input_value(1);

        auto mul_down_input_l_fused =
            mul_down_1_lhs_input == mul_down_1_rhs_input
                ? mul_down_1_lhs_input
                : concat_any(ov::OutputVector{mul_down_1_lhs_input, mul_down_1_rhs_input}, head_axis, rank);
        auto mul_down_input_r_fused =
            mul_down_2_lhs_input == mul_down_2_rhs_input
                ? mul_down_2_lhs_input
                : concat_any(ov::OutputVector{mul_down_2_lhs_input, mul_down_2_rhs_input}, head_axis, rank);

        auto reshape_shape = v0::Constant::create(element::i64,
                                                  Shape{input_fused->get_output_shape(0).size()},
                                                  input_fused->get_output_shape(0));
        auto reshape_fused = reshape_lhs->copy_with_new_inputs({input_fused, reshape_shape});
        copy_runtime_info({reshape_lhs, reshape_rhs}, reshape_fused);
        replace_node(reshape_lhs, reshape_fused);

        auto mul_down_l_fused =
            mul_down_1_lhs->copy_with_new_inputs({mul_down_1_lhs->input_value(0), mul_down_input_l_fused});
        copy_runtime_info({mul_down_1_lhs, mul_down_1_rhs}, mul_down_l_fused);
        replace_node(mul_down_1_lhs, mul_down_l_fused);

        auto mul_down_r_fused =
            mul_down_2_lhs->copy_with_new_inputs({mul_down_2_lhs->input_value(0), mul_down_input_r_fused});
        copy_runtime_info({mul_down_2_lhs, mul_down_2_rhs}, mul_down_r_fused);
        replace_node(mul_down_2_lhs, mul_down_r_fused);

        auto output_node = pm[rope_lhs.output].get_node_shared_ptr();
        replace_node(concat_node, output_node);

        return true;
    });
}

MergeMatMulBiasConcat::MergeMatMulBiasConcat() {
    MATCHER_SCOPE(MergeMatMulBiasConcat);

    struct pattern_nodes {
        std::shared_ptr<Node> convert;
        std::shared_ptr<Node> multiply;
        std::shared_ptr<Node> matmul;
        std::shared_ptr<Node> add;
        std::shared_ptr<Node> output;
    };

    // Helper to create pattern
    auto create_matmul_bias_pattern = [&](const std::shared_ptr<Node>& input) {
        pattern_nodes nodes;
        nodes.convert = pattern::optional<v0::Convert>({pattern::any_input()});
        nodes.multiply = pattern::optional<v1::Multiply>({nodes.convert, pattern::any_input()});
        nodes.matmul = pattern::wrap_type<v0::MatMul>({input, nodes.multiply});
        nodes.add = pattern::wrap_type<v1::Add>({nodes.matmul, pattern::any_input()});
        auto reshape = pattern::wrap_type<v1::Reshape, v0::Unsqueeze>({nodes.add, pattern::any_input()});
        nodes.output = std::make_shared<pattern::op::Or>(OutputVector{reshape, nodes.add});
        return nodes;
    };

    auto input = pattern::any_input();
    auto mm_bias_lhs = create_matmul_bias_pattern(input);
    auto mm_bias_rhs = create_matmul_bias_pattern(input);

    auto concat = pattern::wrap_type<v0::Concat>({mm_bias_lhs.output, mm_bias_rhs.output});
    auto m = std::make_shared<pattern::Matcher>(concat, "MergeMatMulBiasConcat");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = std::dynamic_pointer_cast<v0::Concat>(matcher.get_match_root());
        if (!concat_node || !concat_node->get_rt_info().count("packed_GQA"))
            return false;

        auto mm_lhs = as_type_ptr<v0::MatMul>(pm[mm_bias_lhs.matmul].get_node_shared_ptr());
        auto mm_rhs = as_type_ptr<v0::MatMul>(pm[mm_bias_rhs.matmul].get_node_shared_ptr());
        if (!mm_lhs || !mm_rhs) {
            return false;
        }

        auto add_lhs = as_type_ptr<v1::Add>(pm[mm_bias_lhs.add].get_node_shared_ptr());
        auto add_rhs = as_type_ptr<v1::Add>(pm[mm_bias_rhs.add].get_node_shared_ptr());
        if (!add_lhs || !add_rhs)
            return false;

        // Concatenate along head axis (1)
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]

        std::shared_ptr<ov::Node> input_fused = nullptr;

        auto convert_lhs = ov::as_type_ptr<v0::Convert>(pm[mm_bias_lhs.convert].get_node_shared_ptr());
        auto convert_rhs = ov::as_type_ptr<v0::Convert>(pm[mm_bias_rhs.convert].get_node_shared_ptr());
        if (convert_lhs && convert_rhs) {
            auto convert_const_fused =
                concat_any(OutputVector{convert_lhs->input_value(0), convert_rhs->input_value(0)}, head_axis, rank);
            input_fused = convert_lhs->copy_with_new_inputs({convert_const_fused});
        }

        auto scale_lhs = ov::as_type_ptr<v1::Multiply>(pm[mm_bias_lhs.multiply].get_node_shared_ptr());
        auto scale_rhs = ov::as_type_ptr<v1::Multiply>(pm[mm_bias_rhs.multiply].get_node_shared_ptr());
        if (scale_lhs && scale_rhs) {
            auto scale_fused =
                concat_any(OutputVector{scale_lhs->input_value(1), scale_rhs->input_value(1)}, head_axis, rank);
            if (!input_fused) {
                OutputVector multiply_inputs = {scale_lhs->input_value(0), scale_rhs->input_value(0)};
                input_fused = concat_any(multiply_inputs, head_axis, rank);
            }
            input_fused = scale_lhs->copy_with_new_inputs({input_fused, scale_fused});
        }

        if (!input_fused) {
            input_fused = concat_any(OutputVector{mm_lhs->input_value(1), mm_rhs->input_value(1)}, head_axis, rank);
        }

        auto mm_fused = mm_lhs->copy_with_new_inputs({mm_lhs->input_value(0), input_fused});
        auto bias_fused = concat_any(OutputVector{add_lhs->input_value(1), add_rhs->input_value(1)}, head_axis, rank);
        auto add_fused = add_lhs->copy_with_new_inputs({mm_fused, bias_fused});

        replace_node(concat_node, add_fused);

        return true;
    });
}

MergeDQConcat::MergeDQConcat() {
    MATCHER_SCOPE(MergeDQConcat);

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
        nodes.convert_0 = pattern::wrap_type<v0::Convert>({pattern::wrap_type<v0::Constant>()});
        nodes.convert_1 = pattern::wrap_type<v0::Convert>({pattern::wrap_type<v0::Constant>()});
        nodes.subtract = pattern::wrap_type<v1::Subtract>({nodes.convert_0, nodes.convert_1});
        nodes.multiply = pattern::wrap_type<v1::Multiply>({nodes.subtract, pattern::any_input()});
        nodes.output = pattern::optional<v1::Reshape, v0::Unsqueeze>({nodes.multiply, pattern::any_input()});
        return nodes;
    };

    auto dq_lhs = create_dq_pattern();
    auto dq_rhs = create_dq_pattern();

    auto concat = pattern::wrap_type<v0::Concat>({dq_lhs.output, dq_rhs.output});
    auto m = std::make_shared<pattern::Matcher>(concat, "MergeDQConcat");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = std::dynamic_pointer_cast<v0::Concat>(matcher.get_match_root());
        if (!concat_node || !concat_node->get_rt_info().count("packed_GQA"))
            return false;

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

        // Concatenate along head axis (1)
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]

        auto convert_0_const_fused =
            concat_any(OutputVector{convert_0_lhs->input_value(0), convert_0_rhs->input_value(0)}, head_axis, rank);
        auto convert_0_fused = convert_0_lhs->copy_with_new_inputs({convert_0_const_fused});

        auto convert_1_const_fused =
            concat_any(OutputVector{convert_1_lhs->input_value(0), convert_1_rhs->input_value(0)}, head_axis, rank);
        auto convert_1_fused = convert_1_lhs->copy_with_new_inputs({convert_1_const_fused});

        auto sub_fused = sub_lhs->copy_with_new_inputs({convert_0_fused, convert_1_fused});

        OutputVector mul_inputs_fused = {
            sub_fused,
            concat_any(OutputVector{mul_lhs->input_value(1), mul_rhs->input_value(1)}, head_axis, rank)};

        auto mul_fused = mul_lhs->copy_with_new_inputs(mul_inputs_fused);

        replace_node(concat_node, mul_fused);

        return true;
    });
}

MergeKVCaches::MergeKVCaches() {
    MATCHER_SCOPE(MergeKVCaches);

    // Helper to create cache pattern
    auto create_cache_pattern = [&]() {
        auto cache_input = pattern::any_input();
        auto input = pattern::any_input();
        auto concat = pattern::wrap_type<v0::Concat>({cache_input, input});
        return concat;
    };

    auto cache_lhs = create_cache_pattern();
    auto cache_rhs = create_cache_pattern();

    auto slice_lhs = pattern::wrap_type<v1::StridedSlice>(
        {cache_lhs, pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto slice_rhs = pattern::wrap_type<v1::StridedSlice>(
        {cache_rhs, pattern::any_input(), pattern::any_input(), pattern::any_input()});

    auto concat = pattern::wrap_type<v0::Concat>({cache_lhs, cache_rhs});

    auto m = std::make_shared<pattern::Matcher>(concat, "MergeKVCaches");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        auto pm = matcher.get_pattern_value_map();

        auto concat_node = std::dynamic_pointer_cast<v0::Concat>(matcher.get_match_root());
        if (!concat_node || !concat_node->get_rt_info().count("packed_GQA"))
            return false;

        auto concat_cache_lhs = ov::as_type_ptr<v0::Concat>(pm[cache_lhs].get_node_shared_ptr());
        auto concat_cache_rhs = ov::as_type_ptr<v0::Concat>(pm[cache_rhs].get_node_shared_ptr());
        if (!concat_cache_lhs || !concat_cache_rhs)
            return false;

        // Concatenate along head axis (1)
        int64_t head_axis = 1;
        int64_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]

        auto concat_merged_input1 =
            concat_any(OutputVector{concat_cache_lhs->input_value(0), concat_cache_rhs->input_value(0)},
                       head_axis,
                       rank);
        auto concat_merged_input2 =
            concat_any(OutputVector{concat_cache_lhs->input_value(1), concat_cache_rhs->input_value(1)},
                       head_axis,
                       rank);
        auto concat_merged = concat_cache_lhs->copy_with_new_inputs({concat_merged_input1, concat_merged_input2});

        // spliting cache output
        int64_t len_lhs = concat_node->input_value(0).get_shape()[head_axis];
        int64_t len_rhs = concat_node->input_value(1).get_shape()[head_axis];
        auto axis = v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>({head_axis}));
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