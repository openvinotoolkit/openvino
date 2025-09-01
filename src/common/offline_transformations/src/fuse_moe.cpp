// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe.hpp"

#include <cstdint>
#include <limits>
#include <tuple>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/moe.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"

using namespace ov::gen_pattern;
using namespace ov::pass;

namespace {

auto gen_expert_pattern(std::shared_ptr<ov::Node> final_hidden_states,
                        std::shared_ptr<ov::Node> hidden_states,
                        std::shared_ptr<ov::Node> expert_mask,
                        std::shared_ptr<ov::Node> routing_weights) {
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1));
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

    // TODO(MOE): more patterns
#define WEIGHT_PATTERN(idx)                                                                                   \
    auto weight_const##idx = pattern::wrap_type<ov::op::v0::Constant>();                                      \
    auto weight_const_convert##idx = makePattern<ov::op::v0::Convert>({weight_const##idx});                   \
    auto zp_const##idx = pattern::wrap_type<ov::op::v0::Constant>();                                          \
    auto zp_const_convert##idx = makePattern<ov::op::v0::Convert>({zp_const##idx});                           \
    auto weight_sub_zp##idx =                                                                                 \
        makePattern<ov::op::v1::Subtract>({weight_const_convert##idx, zp_const_convert##idx | zp_const##idx}, \
                                          {{"auto_broadcast", "numpy"}});                                     \
    auto scale_const##idx = pattern::wrap_type<ov::op::v0::Constant>();                                       \
    auto weight_zp##idx = weight_sub_zp##idx | weight_const_convert##idx; /* with zp | w/o zp */              \
    auto weight_mul_scale##idx =                                                                              \
        makePattern<ov::op::v1::Multiply>({weight_zp##idx, scale_const##idx}, {{"auto_broadcast", "numpy"}}); \
    auto weight_mul_scale_reshape##idx =                                                                      \
        makePattern<ov::op::v1::Reshape>({weight_mul_scale##idx, pattern::any_input()});                      \
    auto weight_mul_scale_reshape_convert##idx =                                                              \
        makePattern<ov::op::v0::Convert>({weight_mul_scale_reshape##idx | weight_mul_scale##idx});            \
    /* i4+zp+group+reshape+convert | i4+zp+group+reshape | f16+convert | f32 */                               \
    auto final_weight##idx = weight_mul_scale_reshape_convert##idx | weight_mul_scale_reshape##idx |          \
                             weight_const_convert##idx | weight_const##idx;

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<ov::op::v8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<ov::op::v3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<ov::op::v1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2_0 = makePattern<ov::op::v0::Squeeze>({ListUnpack_Split_2->output(1), 0});
    auto ListUnpack_Squeeze_0_2_1 =
        makePattern<ov::op::v1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_0_2 = ListUnpack_Squeeze_0_2_0 | ListUnpack_Squeeze_0_2_1;
    auto index_add__Convert_2_org =
        makePattern<ov::op::v0::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 =
        makePattern<ov::op::v1::Reshape>({index_add__Convert_2, {-1, 1}}, {{"special_zero", false}});
    auto index_add__Broadcast_25 =
        makePattern<ov::op::v3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_Gather_4 =
        makePattern<ov::op::v8::Gather>({hidden_states, index_add__Convert_2, 1}, {{"batch_dims", 0}});
    auto reshape_Reshape_2 =
        makePattern<ov::op::v1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});
    WEIGHT_PATTERN(0)
    auto gate_linear_MatMul = makePattern<ov::op::v0::MatMul>({reshape_Reshape_2, final_weight0},
                                                              {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = makePattern<ov::op::v4::Swish>({gate_linear_MatMul});
    WEIGHT_PATTERN(1)
    auto up_linear_MatMul = makePattern<ov::op::v0::MatMul>({reshape_Reshape_2, final_weight1},
                                                            {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply =
        makePattern<ov::op::v1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});
    WEIGHT_PATTERN(2)
    auto down_linear_MatMul =
        makePattern<ov::op::v0::MatMul>({mul_Multiply, final_weight2}, {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_2_0 = makePattern<ov::op::v0::Squeeze>({ListUnpack_Split_2->output(0), 0});
    auto ListUnpack_Squeeze_2_1 =
        makePattern<ov::op::v1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_2 = ListUnpack_Squeeze_2_0 | ListUnpack_Squeeze_2_1;
    auto index_Convert_6 = makePattern<ov::op::v0::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<ov::op::v1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split},
                                                              {{"auto_broadcast", "numpy"}});
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<ov::op::v1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2},
                                                    {{"auto_broadcast", "numpy"}});
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<ov::op::v8::Gather>({routing_weights, index_Add_2, 0}, {{"batch_dims", 0}});
    auto index_Reshape_8_2 = makePattern<ov::op::v1::Reshape>({index_Gather_5, {0, 1}}, {{"special_zero", true}});
    auto mul_Multiply_3 = makePattern<ov::op::v1::Multiply>({down_linear_MatMul, index_Gather_5 | index_Reshape_8_2},
                                                            {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_26 =
        makePattern<ov::op::v3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_add__ScatterElementsUpdate_8 = makePattern<ov::op::v12::ScatterElementsUpdate>(
        {final_hidden_states, index_add__Broadcast_25, index_add__Broadcast_26, 0},
        {{"reduction", "sum"}, {"use_init_val", true}});

    auto result = index_add__ScatterElementsUpdate_8;
    auto extract_expert = [=](ov::pass::pattern::Matcher& m,
                              ov::op::v16::MOE::Config& config,
                              std::vector<std::shared_ptr<ov::op::v0::Constant>>& expert_constants) {
        PatternValidator validator(m);
        if (!validator) {
            return -1;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<int>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            return -1;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();

        // Extract constants for this expert
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> gates, ups, downs;

#define GET_MATMUL_PARAM(mat, idx)                                                                              \
    mat[0] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const##idx).get_node_shared_ptr());    \
    if (pattern_map.count(scale_const##idx)) {                                                                  \
        mat[1] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scale_const##idx).get_node_shared_ptr()); \
    }                                                                                                           \
    if (pattern_map.count(zp_const##idx)) {                                                                     \
        mat[2] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zp_const##idx).get_node_shared_ptr());    \
    }

        GET_MATMUL_PARAM(gates, 0)
        GET_MATMUL_PARAM(ups, 1)
        GET_MATMUL_PARAM(downs, 2)
#undef GET_MATMUL_PARAM

        // Store constants in the order: gate_weight, gate_scale, gate_zp, up_weight, up_scale, up_zp, down_weight, down_scale, down_zp
        expert_constants.clear();
        expert_constants.push_back(gates[0]);  // gate_weight
        if (gates[1]) expert_constants.push_back(gates[1]);  // gate_scale (optional)
        if (gates[2]) expert_constants.push_back(gates[2]);  // gate_zp (optional)

        expert_constants.push_back(ups[0]);    // up_weight
        if (ups[1]) expert_constants.push_back(ups[1]);      // up_scale (optional)
        if (ups[2]) expert_constants.push_back(ups[2]);      // up_zp (optional)

        expert_constants.push_back(downs[0]);  // down_weight
        if (downs[1]) expert_constants.push_back(downs[1]);  // down_scale (optional)
        if (downs[2]) expert_constants.push_back(downs[2]);  // down_zp (optional)

        auto gate_shape = gates[0]->get_shape();
        auto up_shape = ups[0]->get_shape();
        auto down_shape = downs[0]->get_shape();
        auto intermediate_size = gate_shape[0];
        size_t group_size = 0;
        // checking weight should be enough, scale/zp should be checked in the pattern
        OPENVINO_ASSERT(gate_shape == up_shape,
                        "up shape must be equal to gate shape, gate shape: ",
                        gate_shape,
                        ", up shape: ",
                        up_shape);
        OPENVINO_ASSERT(hidden_size == down_shape[0],
                        "down weight shape[0] is not expected, expected: ",
                        hidden_size,
                        ", current: ",
                        down_shape[0]);
        if (gate_shape.size() == 3) {
            group_size = gate_shape[2];
            OPENVINO_ASSERT(down_shape.size() == 3 && gate_shape[2] == down_shape[2],
                            "down shape is not compatible gate shape, gate shape: ",
                            gate_shape,
                            ", down shape: ",
                            down_shape);
        }

        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.intermediate_size = intermediate_size;
        config.group_size = group_size;
        config.topk = topk;
        config.weight_type = gates[0]->get_element_type();
        OPENVINO_ASSERT(ups[0]->get_element_type() == config.weight_type,
                        "precision of up weight must be same with gate, gate: ",
                        config.weight_type,
                        ", up: ",
                        ups[0]->get_element_type());
        OPENVINO_ASSERT(downs[0]->get_element_type() == config.weight_type,
                        "precision of down weight must be same with gate, gate: ",
                        config.weight_type,
                        ", down: ",
                        downs[0]->get_element_type());
        if (gates[1]) {
            config.scale_type = gates[1]->get_element_type();
            OPENVINO_ASSERT(ups[1] && ups[1]->get_element_type() == config.scale_type,
                            "precision of up scale must be same with gate, gate: ",
                            config.scale_type,
                            ", up: ",
                            ups[1]->get_element_type());
            OPENVINO_ASSERT(downs[1] && downs[1]->get_element_type() == config.scale_type,
                            "precision of down scale must be same with gate, gate:",
                            config.scale_type,
                            ", down: ",
                            downs[1]->get_element_type());
        }
        if (gates[2]) {
            config.zp_type = gates[2]->get_element_type();
            OPENVINO_ASSERT(ups[2] && ups[2]->get_element_type() == config.zp_type,
                            "precision of up zp must be same with gate, gate: ",
                            config.zp_type,
                            ", up: ",
                            ups[2]->get_element_type());
            OPENVINO_ASSERT(downs[2] && downs[2]->get_element_type() == config.zp_type,
                            "precision of down zp must be same with gate, gate:",
                            config.zp_type,
                            ", down: ",
                            downs[2]->get_element_type());
        }
        return expert_no;
    };

    return std::make_tuple(result, extract_expert);
}

}  // namespace

ov::pass::FuseMOEExpert::FuseMOEExpert() {
    // MATCHER_SCOPE(FuseMOE);

    auto expert_mask = makePattern(ov::Rank(3));
    // shape: [batch * seq_len, hidden_dim]
    // auto final_hidden_states = makePattern<ov::op::v16::MOE>({pattern::any_input(), pattern::any_input()});
    auto final_hidden_states = pattern::wrap_type<ov::op::v16::MOE>();
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3));
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2));
    auto [result, extract_func] = gen_expert_pattern(final_hidden_states, hidden_states, expert_mask, routing_weights);

    matcher_pass_callback callback =
        [OV_CAPTURE_CPY_AND_THIS, result = result, extract_func = extract_func](ov::pass::pattern::Matcher& m) {
            op::v16::MOE::Config config;
            std::vector<std::shared_ptr<ov::op::v0::Constant>> expert_constants;
            auto expert_no = extract_func(m, config, expert_constants);
            if (expert_no < 0)
                return false;

            const auto& pattern_map = m.get_pattern_value_map();
            auto last_node = pattern_map.at(result).get_node_shared_ptr();

            // [final_hidden_states, hidden_states, expert_mask, routing_weights]
            auto prev_moe = pattern_map.at(final_hidden_states).get_node_shared_ptr();
            auto moe = ov::as_type_ptr<op::v16::MOE>(prev_moe);
            OPENVINO_ASSERT(config == moe->get_config(), "each expert config must be same");

            // Create new MOE with additional inputs for this expert
            OutputVector new_args = moe->input_values();

            // Add constants for this expert to the inputs
            for (auto& constant : expert_constants) {
                new_args.push_back(constant);
            }

            auto new_moe = std::make_shared<op::v16::MOE>(new_args, config);
            new_moe->set_friendly_name(moe->get_friendly_name());

            ov::replace_node(last_node, new_moe);
            register_new_node(new_moe);
            return true;
        };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "FuseMOEExpert");
    this->register_matcher(m, callback);
}

ov::pass::FuseMOERouter::FuseMOERouter() {
    // MATCHER_SCOPE(FuseMOERouter);

    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));
    auto router_logits = makePattern(ov::Rank(2));
    // f32[?,128]
    auto softmax_Softmax = makePattern<ov::op::v8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makePattern<ov::op::v11::TopK>({softmax_Softmax, pattern::any_input()});
    topk_TopK->set_output_size(2);
    auto sum_ReduceSum = makePattern<ov::op::v1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = makePattern<ov::op::v1::Divide>({topk_TopK->output(0), sum_ReduceSum},
                                                       {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makePattern<ov::op::v1::OneHot>(
        {topk_TopK->output(1), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makePattern<ov::op::v1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    auto view_Reshape = makePattern(ov::Rank(2));
    // param1: hidden_states: f32[1, -1, 2048]
    auto unsqueeze_Unsqueeze = makePattern<ov::op::v0::Unsqueeze>({view_Reshape, 0});
    auto unsqueeze_Unsqueeze_1 = makePattern<ov::op::v0::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = makePattern<ov::op::v3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Slice = makePattern<ov::op::v8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});
    auto index_ReduceProd = makePattern<ov::op::v1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});
    auto index_Concat = makePattern<ov::op::v0::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});
    // param4: routing weights: [self.topk * batch, 1]
    auto index_Reshape =
        makePattern<ov::op::v1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});

    auto [result, extract_func] =
        gen_expert_pattern(final_hidden_states, unsqueeze_Unsqueeze, permute_Transpose, index_Reshape);

    matcher_pass_callback callback =
        [OV_CAPTURE_CPY_AND_THIS, result = result, extract_func = extract_func](ov::pass::pattern::Matcher& m) {
            op::v16::MOE::Config config;
            std::vector<std::shared_ptr<ov::op::v0::Constant>> expert_constants;
            auto expert_no = extract_func(m, config, expert_constants);
            // must be first expert
            if (expert_no != 0)
                return false;

            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();
            // router_logits: i32[batch*seq, 128]
            auto router_logits_node = pattern_map.at(router_logits).get_node_shared_ptr();
            // f32[batch*seq, 2048]
            auto hidden_states_2d = pattern_map.at(view_Reshape).get_node_shared_ptr();
            // f32[batch*seq, 2048]
            auto last_node = pattern_map.at(result).get_node_shared_ptr();

            OutputVector new_args;
            // hidden_states_2d: f32[batch*seq, 2048]
            // router_logits: i32[batch*seq, 128]
            new_args.push_back(hidden_states_2d);
            new_args.push_back(router_logits_node);

            // Add constants for the first expert
            for (auto& constant : expert_constants) {
                new_args.push_back(constant);
            }

            auto new_node = std::make_shared<op::v16::MOE>(new_args, config);
            // check whether the plugin accepts the config
            if (transformation_callback(new_node)) {
                return false;
            }

            ov::replace_node(last_node, new_node);
            register_new_node(new_node);

            new_node->set_friendly_name("moe_router");

            return true;
        };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "FuseMOERouter");
    this->register_matcher(m, callback);
}

bool ov::pass::FuseMOE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // RUN_ON_MODEL_SCOPE(FuseMOE);
    ov::pass::Manager manager(get_pass_config(), "FuseMOE");

    // manager.register_pass<ov::pass::NopElimination>(true);
    manager.register_pass<ov::pass::EliminateSqueeze>();
    manager.register_pass<FuseMOERouter>();
    manager.register_pass<FuseMOEExpert>();

    manager.run_passes(model);
    return false;
}