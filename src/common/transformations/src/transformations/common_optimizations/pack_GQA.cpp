// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/pack_GQA.hpp"
#include "transformations/common_optimizations/concat_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/block_collection.hpp"
#include "transformations/utils/utils.hpp"
#include "itt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

std::shared_ptr<Node> normalize_rank(const Output<Node>& output, int64_t target_rank) {
    auto pshape = output.get_partial_shape();
    int64_t cur_rank = pshape.rank().is_dynamic() ? 0 : pshape.rank().get_length();
    if (cur_rank >= target_rank)
        return output.get_node_shared_ptr();

    std::vector<int64_t> axes;
    axes.reserve(target_rank - cur_rank);
    for (int64_t i = 0; i < target_rank - cur_rank; ++i)
        axes.push_back(i);

    auto axes_const = ov::op::v0::Constant::create(element::i64, Shape{axes.size()}, axes);
    auto unsqueezed = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(output.get_node_shared_ptr(), axes_const);

    std::cout << "Normalized node: " << unsqueezed->get_friendly_name() << " shape: " << unsqueezed->get_output_partial_shape(0) << std::endl;
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
std::shared_ptr<Node> concat_any(const ov::OutputVector& inputs, int64_t axis = -1, int64_t rank = 0) {
    int64_t max_rank = rank;
    
    for (const auto& t : inputs) {
        auto r = t.get_tensor().get_partial_shape().rank();
        if (r.is_static())
            max_rank = std::max(max_rank, r.get_length());
    }

    OutputVector normalized;
    for (const auto& t : inputs) {
        normalized.push_back(normalize_rank(t, max_rank));
        std::cout << "Input node for concat: " << t.get_node()->get_friendly_name() << " shape: " << normalized.back().get_node()->get_output_partial_shape(0) << std::endl;
    }
    
    auto concat = ov::op::util::make_try_fold<v0::Concat>(normalized, axis);
    
    std::cout << "Concat node: " << concat->get_friendly_name() << " shape: " << concat->get_output_partial_shape(0) << std::endl;

    return concat;
}

}  // namespace

bool PackGQA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PackGQA);
    ov::pass::Manager manager(get_pass_config(), "PackGQA");
    
    manager.register_pass<ov::pass::Serialize>("PackGQA_before.xml", "PackGQA_before.bin");
    manager.register_pass<ov::pass::MergeTwoUnrolledSDPAAdd>(); 
    manager.register_pass<ov::pass::Serialize>("PackGQA_sdpa_fused.xml", "PackGQA_sdpa_fused.bin");
    manager.register_pass<ov::pass::MergeTwoUnrolledRoPEConcat>();
    manager.register_pass<ov::pass::Serialize>("PackGQA_after.xml", "PackGQA_after.bin");
    manager.register_pass<ov::pass::ConcatFusion>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::Serialize>("PackGQA_after_folding.xml", "PackGQA_after_folding.bin");
    
    return manager.run_passes(model);
}

// Helper to skip optional nodes
template<typename T>
std::shared_ptr<ov::Node> skip_node(const std::shared_ptr<ov::Node>& node) {
    if (auto reduce = as_type_ptr<T>(node)) {
        return reduce->input_value(0).get_node_shared_ptr();
    }
    return node;
};

        
// Helper function to extract scale node (supports both Divide and Multiply)
static std::shared_ptr<Node> get_scale(const std::shared_ptr<ov::Node>& bias_node) {
    auto input_node = bias_node->input_value(0).get_node_shared_ptr();
    if (auto div = as_type_ptr<v1::Divide>(input_node)) {
        return div;
    }
    if (auto mul = as_type_ptr<v1::Multiply>(input_node)) {
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
        auto q = any_input(); 
        auto k = any_input();
        auto v = any_input();
        auto k_scale = optional<v1::Multiply>({k, any_input()});
        auto qk = wrap_type<v0::MatMul>({q, k_scale});
        auto qk_scale = optional<v1::Divide,v1::Multiply>({qk, any_input()});
        auto bias_add = wrap_type<v1::Add>({qk_scale, any_input()});
        auto softmax = wrap_type<v8::Softmax>({bias_add});
        auto qkv = wrap_type<v0::MatMul>({softmax, v});
        auto qkv_reshaped = optional<v1::Reshape>({qkv, any_input()});
        auto proj = any_input();
        auto matmul = wrap_type<v0::MatMul>({qkv_reshaped, proj});
        auto matmul_reduced = optional<v1::ReduceSum>({matmul, any_input()});
        return matmul_reduced;
    };
    
    auto sdpa_lhs = create_sdpa_pattern();
    auto sdpa_rhs = create_sdpa_pattern();
    
    auto add = wrap_type<v1::Add>({sdpa_lhs, sdpa_rhs});

    auto m = std::make_shared<pattern::Matcher>(add, "MergeTwoUnrolledSDPAAdd");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        std::cout << "MergeTwoUnrolledSDPAAdd transformation is started." << std::endl;
        
        auto add_node = std::dynamic_pointer_cast<v1::Add>(matcher.get_match_root());
        if (!add_node)
            return false;

        // Extract MatMul nodes
        auto lhs = skip_node<v1::ReduceSum>(add_node->input_value(0).get_node_shared_ptr());
        auto rhs = skip_node<v1::ReduceSum>(add_node->input_value(1).get_node_shared_ptr());
        
        auto mm1 = as_type_ptr<v0::MatMul>(lhs);
        auto mm2 = as_type_ptr<v0::MatMul>(rhs);
        if (!mm1 || !mm2)
            return false;

        auto proj1 = mm1->input_value(1);
        auto proj2 = mm2->input_value(1);
        
        // Get SDPA MatMul nodes (QKV)
        auto sdpa_mm1 = as_type_ptr<v0::MatMul>(skip_node<v1::Reshape>(mm1->input_value(0).get_node_shared_ptr()));
        auto sdpa_mm2 = as_type_ptr<v0::MatMul>(skip_node<v1::Reshape>(mm2->input_value(0).get_node_shared_ptr()));
        if (!sdpa_mm1 || !sdpa_mm2)
            return false;

        // Extract Softmax nodes
        auto soft1 = as_type_ptr<v8::Softmax>(sdpa_mm1->input_value(0).get_node_shared_ptr());
        auto soft2 = as_type_ptr<v8::Softmax>(sdpa_mm2->input_value(0).get_node_shared_ptr());
        if (!soft1 || !soft2)
            return false;
        
        // Extract bias Add nodes
        auto bias1 = as_type_ptr<v1::Add>(soft1->input_value(0).get_node_shared_ptr());
        auto bias2 = as_type_ptr<v1::Add>(soft2->input_value(0).get_node_shared_ptr());
        if (!bias1 || !bias2)
            return false;
        
        // Extract optional scale nodes (Divide or Multiply)
        auto sf1 = get_scale(bias1);
        auto sf2 = get_scale(bias2);
        
        auto qk1_node = sf1 ? sf1->input_value(0).get_node_shared_ptr() : bias1->input_value(0).get_node_shared_ptr();
        auto qk2_node = sf2 ? sf2->input_value(0).get_node_shared_ptr() : bias2->input_value(0).get_node_shared_ptr();
        
        auto qk1 = as_type_ptr<v0::MatMul>(qk1_node);
        auto qk2 = as_type_ptr<v0::MatMul>(qk2_node);
        if (!qk1 || !qk2)
            return false;

        // Verify scales match if present
        if (sf1 && sf2) {
            auto c1 = as_type_ptr<v0::Constant>(sf1->input_value(1).get_node_shared_ptr());
            auto c2 = as_type_ptr<v0::Constant>(sf2->input_value(1).get_node_shared_ptr());
            if (!c1 || !c2 || c1->cast_vector<float>() != c2->cast_vector<float>())
            return false;
        }

        // Extract Q, K, V, P, B from both SDPAs
        Output<Node> Q1 = qk1->input_value(0), K1 = qk1->input_value(1);
        Output<Node> V1 = sdpa_mm1->input_value(1), P1 = proj1, B1 = bias1->input_value(1);
        
        Output<Node> Q2 = qk2->input_value(0), K2 = qk2->input_value(1);
        Output<Node> V2 = sdpa_mm2->input_value(1), P2 = proj2, B2 = bias2->input_value(1);

        // Concatenate along head axis (1)
        std::cout << "Concatenating Q, K, V, B, P tensors..." << std::endl;
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]
        auto Q = concat_any(OutputVector{Q1, Q2}, head_axis, rank);
        auto K = concat_any(OutputVector{K1, K2}, head_axis, rank);
        auto V = concat_any(OutputVector{V1, V2}, head_axis, rank);
        auto B = (B1 == B2) ? B1 : concat_any(OutputVector{B1, B2}, head_axis, rank);
        auto P = concat_any(OutputVector{P1, P2}, head_axis, rank);

        // Rebuild fused SDPA
        auto qk_fused = std::make_shared<v0::MatMul>(Q, K);
        std::shared_ptr<ov::Node> scores_scaled = qk_fused;
        
        if (sf1 && sf2) {
            scores_scaled = sf1->copy_with_new_inputs({qk_fused, sf1->input_value(1)});
        }
        
        auto bias_fused = std::make_shared<v1::Add>(scores_scaled, B);
        auto softmax_fused = std::make_shared<v8::Softmax>(bias_fused, -1);
        auto qkv_fused = sdpa_mm1->copy_with_new_inputs({softmax_fused, V});
        auto proj = mm1->copy_with_new_inputs({qkv_fused, P});
        
        auto reduce_axis = v0::Constant::create(element::i64, Shape{1}, {1});
        auto reduce = std::make_shared<v1::ReduceSum>(proj, reduce_axis, false);

        reduce->set_friendly_name(add_node->get_friendly_name() + "_reduced");
        copy_runtime_info({add_node, mm1, mm2, soft1, soft2}, reduce);
        
        std::cout << "Replacing Add with fused ReduceSum: " << reduce->get_output_partial_shape(0) << std::endl;
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
        std::shared_ptr<Node> output;
    };
    
    // Helper to create SDPA pattern
    auto create_rope_pattern = [&]() {
        pattern_nodes nodes;
        auto input = any_input();
        nodes.input = wrap_type<v1::Reshape>({input, any_input()});
        auto var_split = wrap_type<v1::VariadicSplit>({nodes.input, any_input(), any_input()});
        var_split->set_output_size(2);
        
        nodes.scale = wrap_type<v0::Negative>({var_split->output(1)});
        auto concat = wrap_type<v0::Concat>({nodes.scale, var_split->output(0)});
        nodes.mul_l= wrap_type<v1::Multiply>({concat, any_input()});
        nodes.mul_r = wrap_type<v1::Multiply>({nodes.input, any_input()});
        nodes.output = wrap_type<v1::Add>({nodes.mul_r, nodes.mul_l});  // todo: use mul_2 as 2nd input
        return nodes;
    };
    
    auto rope_lhs = create_rope_pattern();
    auto rope_rhs = create_rope_pattern();
    
    auto concat = wrap_type<v0::Concat>({rope_lhs.output, rope_rhs.output});

    auto m = std::make_shared<pattern::Matcher>(concat, "MergeTwoUnrolledRoPEConcat");
    register_matcher(m, [=](pattern::Matcher& matcher) {
        std::cout << "MergeTwoUnrolledRoPEConcat transformation is started." << std::endl;
        
        auto pm = matcher.get_pattern_value_map();
        
        auto concat_node = std::dynamic_pointer_cast<v0::Concat>(matcher.get_match_root());
        if (!concat_node)
            return false;
        
        auto add_lhs = as_type_ptr<v1::Add>(pm[rope_lhs.output].get_node_shared_ptr());
        auto add_rhs = as_type_ptr<v1::Add>(pm[rope_rhs.output].get_node_shared_ptr());
        if (!add_lhs || !add_rhs)
            return false;
            
        // Get down Multiply nodes
        auto mul_down_1_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_lhs = as_type_ptr<v1::Multiply>(pm[rope_lhs.mul_r].get_node_shared_ptr());
        auto mul_down_1_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_l].get_node_shared_ptr());
        auto mul_down_2_rhs = as_type_ptr<v1::Multiply>(pm[rope_rhs.mul_r].get_node_shared_ptr());
        if (!mul_down_1_lhs || !mul_down_2_lhs || !mul_down_1_rhs || !mul_down_2_rhs)
            return false;
        
        // Extract optional scale nodes (Divide or Multiply)
        auto sf_lhs = as_type_ptr<v0::Negative>(pm[rope_lhs.scale].get_node_shared_ptr());
        auto sf_rhs = as_type_ptr<v0::Negative>(pm[rope_rhs.scale].get_node_shared_ptr());
        
        // extract inputs
        auto reshape_lhs = pm[rope_lhs.input].get_node_shared_ptr();
        auto reshape_rhs = pm[rope_rhs.input].get_node_shared_ptr();
        
        // Concatenate along head axis (1)
        std::cout << "Concatenating Input tensors" << std::endl;
        size_t head_axis = 1;
        size_t rank = 4;  // Assuming 4D tensors [batch, heads, seq_len, dim]
        
        auto input_fused = concat_any(OutputVector{reshape_lhs->input_value(0), reshape_rhs->input_value(0)}, head_axis, rank);
        
        auto mul_down_1_lhs_input = mul_down_1_lhs->input_value(1);
        auto mul_down_2_lhs_input = mul_down_2_lhs->input_value(1);
        auto mul_down_1_rhs_input = mul_down_1_rhs->input_value(1);
        auto mul_down_2_rhs_input = mul_down_2_rhs->input_value(1);
        
        auto mul_down_input_l_fused = mul_down_1_lhs_input == mul_down_1_rhs_input ? mul_down_1_lhs_input : concat_any(OutputVector{mul_down_1_lhs_input, mul_down_1_rhs_input}, head_axis, rank);
        auto mul_down_input_r_fused = mul_down_2_lhs_input == mul_down_2_rhs_input ? mul_down_2_lhs_input : concat_any(OutputVector{mul_down_2_lhs_input, mul_down_2_rhs_input}, head_axis, rank);
        
        std::cout << "Replace input node: " << reshape_lhs->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name() << " shape: " << std::endl;
        auto reshape_shape = op::v0::Constant::create(element::i64, Shape{input_fused->get_output_shape(0).size()}, input_fused->get_output_shape(0));
        auto reshape_fused = reshape_lhs->copy_with_new_inputs({input_fused, reshape_shape});
        replace_node(reshape_lhs, reshape_fused);
        
        std::cout << "Replace input node: " << mul_down_1_lhs->get_input_source_output(1).get_node_shared_ptr()->get_friendly_name() << " shape: " << std::endl;
        auto mul_down_l_fused = mul_down_1_lhs->copy_with_new_inputs({mul_down_1_lhs->input_value(0), mul_down_input_l_fused});
        replace_node(mul_down_1_lhs, mul_down_l_fused);
        
        std::cout << "Replace input node: " << mul_down_2_lhs->get_input_source_output(1).get_node_shared_ptr()->get_friendly_name() << " shape: " << std::endl;
        auto mul_down_r_fused = mul_down_2_lhs->copy_with_new_inputs({mul_down_2_lhs->input_value(0), mul_down_input_r_fused});
        replace_node(mul_down_2_lhs, mul_down_r_fused);
        
        replace_node(concat_node, add_lhs);
        return true;
    });
}