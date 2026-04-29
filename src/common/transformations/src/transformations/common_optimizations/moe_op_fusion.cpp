// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moe_op_fusion.hpp"

#include <limits>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::pass {

using ov::op::internal::GatherMatmul;
using ov::op::internal::GatherMatmulCompressed;
using ov::op::internal::MOECompressed;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;

Convert3GatherMatmulMoeBlockToMoeOp::Convert3GatherMatmulMoeBlockToMoeOp(size_t has_batch_dim) {
    MATCHER_SCOPE(Convert3GatherMatmulMoeBlockToMoeOp);

    auto experts_reshape_m = pattern::any_input();
    auto unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({experts_reshape_m, pattern::any_input()});

    auto gate_w_m = pattern::any_input();
    auto topk_indices_m = pattern::any_input();

    // Plain BGM (4 inputs: A, B, indices, bias)
    auto bgm_gate_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, gate_w_m, topk_indices_m, pattern::any_input()});
    // Compressed BGM (6 inputs: A, B, indices, bias, scale, zp)
    auto gate_scale_m = pattern::any_input();
    auto gate_zp_m = pattern::any_input();
    auto bgm_gate_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, gate_w_m, topk_indices_m, pattern::any_input(), gate_scale_m, gate_zp_m});
    // Or-pattern
    auto bgm_gate_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_gate_4_m, bgm_gate_6_m});

    auto swish_m = pattern::wrap_type<v4::Swish>({bgm_gate_m});

    auto up_w_m = pattern::any_input();
    auto bgm_up_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, up_w_m, topk_indices_m, pattern::any_input()});
    auto up_scale_m = pattern::any_input();
    auto up_zp_m = pattern::any_input();
    auto bgm_up_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, up_w_m, topk_indices_m, pattern::any_input(), up_scale_m, up_zp_m});
    auto bgm_up_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_up_4_m, bgm_up_6_m});

    auto swiglu_m = pattern::wrap_type<v1::Multiply>({swish_m, bgm_up_m});

    auto down_w_m = pattern::any_input();
    auto bgm_down_4_m = pattern::wrap_type<GatherMatmul>({swiglu_m, down_w_m, topk_indices_m, pattern::any_input()});
    auto down_scale_m = pattern::any_input();
    auto down_zp_m = pattern::any_input();
    auto bgm_down_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {swiglu_m, down_w_m, topk_indices_m, pattern::any_input(), down_scale_m, down_zp_m});
    auto bgm_down_m = std::make_shared<pattern::op::Or>(OutputVector{bgm_down_4_m, bgm_down_6_m});

    // Compact routing: Transpose → Unsqueeze
    auto routing_transpose_m = pattern::wrap_type<v1::Transpose>({pattern::any_input(), pattern::any_input()});
    auto routing_unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({routing_transpose_m, pattern::any_input()});

    auto final_mul_m = pattern::wrap_type<v1::Multiply>({bgm_down_m, routing_unsqueeze_m}, pattern::consumers_count(1));
    auto reduce_sum_m = pattern::wrap_type<v1::ReduceSum>({final_mul_m, pattern::any_input()}, {{"keep_dims", false}});
    auto end_reshape_shape_m = pattern::any_input();
    auto end_reshape_m = pattern::wrap_type<v1::Reshape>({reduce_sum_m, end_reshape_shape_m});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto experts_reshape_node = pm.at(experts_reshape_m).get_node_shared_ptr();
        auto hidden_states = experts_reshape_node->input_value(0);

        auto routing = pm.at(routing_unsqueeze_m).get_node_shared_ptr();
        auto topk_indices = pm.at(topk_indices_m);
        auto gate_w = pm.at(gate_w_m);
        auto up_w = pm.at(up_w_m);
        auto down_w = pm.at(down_w_m);

        // Extract expert_beta from Swish
        float expert_beta = 1.0f;
        auto swish_node = pm.at(swish_m).get_node_shared_ptr();

        auto swish_op = ov::as_type_ptr<v4::Swish>(swish_node);
        OPENVINO_ASSERT(swish_op, "Unexpected node type matched for Swish: ", *swish_node);

        if (swish_op->get_input_size() > 1) {
            if (auto beta_const = ov::as_type_ptr<v0::Constant>(swish_op->get_input_node_shared_ptr(1))) {
                expert_beta = beta_const->cast_vector<float>()[0];
            }
        }

        std::shared_ptr<ov::Node> moe_node;
        const bool is_gate_compressed = pm.count(bgm_gate_6_m);
        const bool is_up_compressed = pm.count(bgm_up_6_m);
        const bool is_down_compressed = pm.count(bgm_down_6_m);

        // Bail out if BGMs are mixed (some compressed, some plain). The current
        // MOE/MOECompressed configs assume all-or-nothing compression.
        if ((is_gate_compressed != is_up_compressed) || (is_gate_compressed != is_down_compressed)) {
            return false;
        }

        const bool is_compressed = is_gate_compressed;

        if (is_compressed) {
            // Build MOECompressed with 12 inputs: hidden, routing, topk,
            // gate_w, gate_scale, gate_zp, up_w, up_scale, up_zp, down_w, down_scale, down_zp
            ov::OutputVector moe_inputs = {
                hidden_states,
                routing,
                topk_indices,
                gate_w,
                pm.at(gate_scale_m),
                pm.at(gate_zp_m),
                up_w,
                pm.at(up_scale_m),
                pm.at(up_zp_m),
                down_w,
                pm.at(down_scale_m),
                pm.at(down_zp_m),
            };

            // Populate compressed config from weight shapes
            auto wei_partial_shape = gate_w.get_partial_shape();
            OPENVINO_ASSERT(wei_partial_shape.is_static(), "MOE weight shape should be static.");
            auto weight_shape = wei_partial_shape.to_shape();
            bool group_compressed = (weight_shape.size() == 4);

            auto topk_shape = topk_indices.get_partial_shape();
            OPENVINO_ASSERT(topk_shape[1].is_static(), "K dimension in moe topk input should be static.");

            MOECompressed::Config compressed_config{
                {ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, 0.0f, expert_beta},
                group_compressed ? weight_shape[2] * weight_shape[3] : weight_shape[2],
                weight_shape[1],
                weight_shape[0],
                0,  // num_shared_expert
                static_cast<size_t>(topk_shape[1].get_length()),
                group_compressed ? weight_shape[3] : std::numeric_limits<size_t>::max(),
                has_batch_dim,
                false,
                ov::element::f16,
            };

            auto moe_compressed = std::make_shared<MOECompressed>(moe_inputs, compressed_config);

            // Insert Convert if output type was forced and differs from original
            if (moe_compressed->get_output_element_type(0) != hidden_states.get_element_type()) {
                moe_compressed->set_friendly_name(m.get_match_root()->get_friendly_name() + "/MOECompressed");
                auto convert = std::make_shared<v0::Convert>(moe_compressed, hidden_states.get_element_type());
                convert->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), {moe_compressed, convert});
                moe_node = convert;
            } else {
                moe_node = moe_compressed;
            }
        } else {
            ov::op::internal::MOE::Config config{ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU, 0.0f, expert_beta};
            // Plain MOE with 6 inputs
            ov::OutputVector moe_inputs = {hidden_states, routing, topk_indices, gate_w, up_w, down_w};

            moe_node = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        }

        moe_node->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe_node);
        ov::replace_node(m.get_match_root(), moe_node);

        register_new_node(moe_node);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(end_reshape_m, matcher_name);
    this->register_matcher(matcher, callback);
}

Convert2GatherMatmulMoeBlockToMoeOp::Convert2GatherMatmulMoeBlockToMoeOp(size_t has_batch_dim) {
    MATCHER_SCOPE(Convert2GatherMatmulMoeBlockToMoeOp);

    auto experts_reshape_m = pattern::any_input();
    auto unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({experts_reshape_m, pattern::any_input()});

    auto gate_up_w_m = pattern::any_input();
    auto topk_indices_m = pattern::any_input();
    auto gate_up_bias_m = pattern::any_input();

    // Plain BGM (4 inputs)
    auto bgm_gate_up_4_m = pattern::wrap_type<GatherMatmul>({unsqueeze_m, gate_up_w_m, topk_indices_m, gate_up_bias_m});
    // Compressed BGM (6 inputs)
    auto gate_up_scale_m = pattern::any_input();
    auto gate_up_zp_m = pattern::any_input();
    auto bgm_gate_up_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {unsqueeze_m, gate_up_w_m, topk_indices_m, gate_up_bias_m, gate_up_scale_m, gate_up_zp_m});
    auto bgm_gate_up_m = bgm_gate_up_4_m | bgm_gate_up_6_m;

    // Activation subgraph between gate_up and down BGMs
    auto slice1_m = pattern::wrap_type<v8::Slice>(
        {bgm_gate_up_m, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto clamp_m = pattern::wrap_type<v0::Clamp>({slice1_m});
    auto add1_m = pattern::wrap_type<v1::Add>({clamp_m, pattern::wrap_const()});

    auto slice2_m = pattern::wrap_type<v8::Slice>(
        {bgm_gate_up_m, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto minimum1_m = pattern::wrap_type<v1::Minimum>({slice2_m, pattern::wrap_const()});
    auto swish_beta_m = pattern::wrap_const();
    auto swish_m = pattern::wrap_type<v4::Swish>({minimum1_m, swish_beta_m});

    auto multiply2_m = pattern::wrap_type<v1::Multiply>({add1_m, swish_m});

    auto down_w_m = pattern::any_input();
    auto down_bias_m = pattern::any_input();
    // Plain BGM (4 inputs)
    auto bgm_down_4_m = pattern::wrap_type<GatherMatmul>({multiply2_m, down_w_m, topk_indices_m, down_bias_m});
    // Compressed BGM (6 inputs)
    auto down_scale_m = pattern::any_input();
    auto down_zp_m = pattern::any_input();
    auto bgm_down_6_m = pattern::wrap_type<GatherMatmulCompressed>(
        {multiply2_m, down_w_m, topk_indices_m, down_bias_m, down_scale_m, down_zp_m});
    auto bgm_down_m = bgm_down_4_m | bgm_down_6_m;

    // Compact routing: Transpose → Unsqueeze
    auto routing_transpose_m = pattern::wrap_type<v1::Transpose>({pattern::any_input(), pattern::any_input()});
    auto routing_unsqueeze_m = pattern::wrap_type<v0::Unsqueeze>({routing_transpose_m, pattern::any_input()});

    auto final_mul_m = pattern::wrap_type<v1::Multiply>({bgm_down_m, routing_unsqueeze_m});
    auto reduce_sum_m = pattern::wrap_type<v1::ReduceSum>({final_mul_m, pattern::any_input()}, {{"keep_dims", false}});
    auto end_reshape_shape_m = pattern::any_input();
    auto end_reshape_m = pattern::wrap_type<v1::Reshape>({reduce_sum_m, end_reshape_shape_m});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pm = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto experts_reshape_node = pm.at(experts_reshape_m).get_node_shared_ptr();
        auto hidden_states = experts_reshape_node->input_value(0);

        auto routing = pm.at(routing_unsqueeze_m).get_node_shared_ptr();
        auto topk_indices = pm.at(topk_indices_m);
        auto gate_up_w = pm.at(gate_up_w_m);
        auto gate_up_bias = pm.at(gate_up_bias_m);
        auto down_w = pm.at(down_w_m);
        auto down_bias = pm.at(down_bias_m);

        // Extract expert_beta from Swish beta
        auto swish_beta_const = ov::as_type_ptr<v0::Constant>(pm.at(swish_beta_m).get_node_shared_ptr());
        float expert_beta = swish_beta_const->cast_vector<float>()[0];

        // Extract expert_alpha from Clamp max
        float expert_alpha = 0.0f;

        auto clamp_node = pm.at(clamp_m).get_node_shared_ptr();
        auto clamp_op = ov::as_type_ptr<v0::Clamp>(clamp_node);
        OPENVINO_ASSERT(clamp_op, "Unexpected node type matched for clamp: ", *clamp_node);

        std::shared_ptr<ov::Node> moe_node;

        const bool is_gate_up_compressed = pm.count(bgm_gate_up_6_m) > 0;
        const bool is_down_compressed = pm.count(bgm_down_6_m) > 0;

        // Bail out if BGMs are mixed (some compressed, some plain).
        if (is_gate_up_compressed != is_down_compressed) {
            return false;
        }

        const bool is_compressed = is_gate_up_compressed;
        if (is_compressed) {
            // Build MOECompressed inputs
            // GEMM2 compressed layout: hidden, routing, topk,
            // gate_up_w, gate_up_scale, [gate_up_zp,] gate_up_bias,
            // down_w, down_scale, [down_zp,] down_bias
            ov::OutputVector moe_inputs;
            moe_inputs.push_back(hidden_states);
            moe_inputs.push_back(routing);
            moe_inputs.push_back(topk_indices);

            // Absent zp is represented as Constant(element::dynamic, Shape{0})
            bool has_zp = pm.at(gate_up_zp_m).get_element_type() != ov::element::dynamic;

            // gate_up params
            moe_inputs.push_back(gate_up_w);
            moe_inputs.push_back(pm.at(gate_up_scale_m));
            if (has_zp) {
                moe_inputs.push_back(pm.at(gate_up_zp_m));
            }
            moe_inputs.push_back(gate_up_bias);

            // down params
            moe_inputs.push_back(down_w);
            moe_inputs.push_back(pm.at(down_scale_m));
            if (has_zp) {
                moe_inputs.push_back(pm.at(down_zp_m));
            }
            moe_inputs.push_back(down_bias);

            // Populate compressed config from weight shapes
            auto weight_shape = gate_up_w.get_shape();
            auto scale_shape = pm.at(gate_up_scale_m).get_shape();
            bool group_compressed = (weight_shape.size() == 4);
            size_t hidden = group_compressed ? weight_shape[2] * weight_shape[3] : weight_shape[2];

            auto topk_indices_shape = topk_indices.get_partial_shape();
            auto topk_rank = topk_indices_shape.rank().get_length();
            OPENVINO_ASSERT(topk_indices_shape[topk_rank - 1].is_static(),
                            "K dimension in moe topk_indices input should be static.");

            MOECompressed::Config compressed_config{
                {ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP, expert_alpha, expert_beta},
                hidden,
                weight_shape[1],
                weight_shape[0],
                0,  // num_shared_expert
                static_cast<size_t>(topk_indices_shape[topk_rank - 1].get_length()),
                group_compressed ? weight_shape[3] : std::numeric_limits<size_t>::max(),
                has_batch_dim,
                has_zp,
                ov::element::dynamic,
            };

            moe_node = std::make_shared<MOECompressed>(moe_inputs, compressed_config);
        } else {
            const ov::op::internal::MOE::Config config{ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP,
                                                       expert_alpha,
                                                       expert_beta};
            // Plain MOE with 7 inputs
            const ov::OutputVector moe_inputs =
                {hidden_states, routing, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

            moe_node = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
        }

        moe_node->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe_node);
        ov::replace_node(m.get_match_root(), moe_node);

        register_new_node(moe_node);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(end_reshape_m, matcher_name);
    this->register_matcher(matcher, callback);
}

}  // namespace ov::pass
