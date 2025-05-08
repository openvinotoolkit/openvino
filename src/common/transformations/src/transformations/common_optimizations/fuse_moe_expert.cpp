// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_moe_expert.hpp"

#include <cstdint>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/moe_expert.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/pattern/op/optional.hpp"

using namespace ov::gen_pattern;
using namespace ov::pass;

ov::pass::FuseMoeExpert::FuseMoeExpert() {
    MATCHER_SCOPE(FuseMoeExpert);

    auto expert_mask = makePattern(ov::Rank(3));
    // shape: [batch * seq_len, hidden_dim]
    auto final_hidden_states = makePattern(ov::Rank(2));
    // shape: [1, batch * seq_len, hidden_dim]
    auto hidden_states = makePattern(ov::Rank(3));
    // shape: [1], aka topk
    auto routing_weights_shapeof_split = makePattern(ov::Rank(1));
    // shape: [self.topk * batch, 1]
    auto routing_weights = makePattern(ov::Rank(2));
    // shape: [2], data = [1, hidden_size]
    auto index_add__ShapeOf_22 = makePattern("[2]");

    auto hidden_size = ov::gen_pattern::Symbol("hidden_size");
    auto expert_no = ov::gen_pattern::Symbol("expert_no");

#define WEIGHT_PATTERN(idx) \
    auto weight_const##idx = pattern::wrap_type<ov::op::v0::Constant>();  \
    auto weight_const_convert##idx = makePattern<opset1::Convert>({weight_const##idx}); \
    auto zp_const##idx = pattern::wrap_type<ov::op::v0::Constant>();  \
    auto zp_const_convert##idx = makePattern<opset1::Convert>({zp_const##idx}); \
    auto weight_sub_zp##idx = makePattern<opset1::Subtract>({weight_const_convert##idx, zp_const_convert##idx | zp_const##idx}, {{"auto_broadcast", "numpy"}});   \
    auto scale_const##idx = pattern::wrap_type<ov::op::v0::Constant>();   \
    auto weight_zp##idx = weight_sub_zp##idx | weight_const_convert##idx; /* with zp | w/o zp */  \
    auto weight_mul_scale##idx = makePattern<opset1::Multiply>({weight_sub_zp##idx, scale_const##idx}, {{"auto_broadcast", "numpy"}});    \
    auto weight_mul_scale_reshape##idx = makePattern<opset1::Reshape>({weight_mul_scale##idx, pattern::any_input()});   \
    auto weight_mul_scale_reshape_convert##idx = makePattern<opset1::Convert>({weight_mul_scale_reshape##idx});     \
    /* i4+zp+group+convert | i4+zp+group | i4+zp | f16+convert | f32 */     \
    auto final_weight##idx = weight_mul_scale_reshape_convert##idx | weight_mul_scale_reshape##idx; //weight_mul_scale##idx | weight_mul_scale##idx | weight_const_convert##idx | weight_const##idx;

    // expert_mask[expert_idx]
    auto select_Gather_2 = makePattern<opset8::Gather>({expert_mask, expert_no, 0}, {{"batch_dims", 0}});
    // x = torch.where(expert_mask[expert_idx]), x shape: [2, nonzero], dim0: topk, dim1: batch
    auto ListUnpack_NonZero_2 = makePattern<opset3::NonZero>({select_Gather_2}, {{"output_type", "i64"}});
    // topk, batch = torch.where(expert_mask[expert_idx])
    auto ListUnpack_Split_2 = makePattern<opset1::Split>({ListUnpack_NonZero_2, 0}, {{"num_splits", 2}});
    ListUnpack_Split_2->set_output_size(2);
    // batch
    auto ListUnpack_Squeeze_0_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(1), 0});
    auto ListUnpack_Squeeze_0_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(1), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_0_2 = ListUnpack_Squeeze_0_2_0 | ListUnpack_Squeeze_0_2_1;
    auto index_add__Convert_2_org = makePattern<opset1::Convert>({ListUnpack_Squeeze_0_2}, {{"destination_type", "i32"}});
    auto index_add__Convert_2 = index_add__Convert_2_org | ListUnpack_Squeeze_0_2;
    auto index_add__Reshape_2 = makePattern<opset1::Reshape>({index_add__Convert_2, {-1,1}}, {{"special_zero", false}});
    auto index_add__Broadcast_25 = makePattern<opset3::Broadcast>({index_add__Reshape_2, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_Gather_4 = makePattern<opset8::Gather>({hidden_states, index_add__Convert_2, 1}, {{"batch_dims", 0}});
    auto reshape_Reshape_2 = makePattern<opset1::Reshape>({index_Gather_4, {-1, hidden_size}}, {{"special_zero", true}});
    WEIGHT_PATTERN(0)
    auto gate_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, final_weight0}, {{"transpose_a", false}, {"transpose_b", true}});
    auto silu_Swish = makePattern<opset4::Swish>({gate_linear_MatMul});
    WEIGHT_PATTERN(1)
    auto up_linear_MatMul = makePattern<opset1::MatMul>({reshape_Reshape_2, final_weight1}, {{"transpose_a", false}, {"transpose_b", true}});
    auto mul_Multiply = makePattern<opset1::Multiply>({silu_Swish, up_linear_MatMul}, {{"auto_broadcast", "numpy"}});
    WEIGHT_PATTERN(2)
    auto down_linear_MatMul = makePattern<opset1::MatMul>({mul_Multiply, final_weight2}, {{"transpose_a", false}, {"transpose_b", true}});
    auto ListUnpack_Squeeze_2_0 = makePattern<opset1::Squeeze>({ListUnpack_Split_2->output(0), 0});
    auto ListUnpack_Squeeze_2_1 = makePattern<opset1::Reshape>({ListUnpack_Split_2->output(0), {-1}}, {{"special_zero", false}});
    auto ListUnpack_Squeeze_2 = ListUnpack_Squeeze_2_0 | ListUnpack_Squeeze_2_1;
    auto index_Convert_6 = makePattern<opset1::Convert>({ListUnpack_Squeeze_2}, {{"destination_type", "i32"}});
    // self.topk * batch, index_split=shapeof(routing_weights), shape: [batch, self.topk, 1]
    auto index_Multiply_2 = makePattern<opset1::Multiply>({index_add__Convert_2, routing_weights_shapeof_split}, {{"auto_broadcast", "numpy"}});
    // self.topk * batch + topk
    auto index_Add_2 = makePattern<opset1::Add>({index_Convert_6 | ListUnpack_Squeeze_2, index_Multiply_2}, {{"auto_broadcast", "numpy"}});
    // routing_weights', shape[self.topk * batch, 1]
    auto index_Gather_5 = makePattern<opset8::Gather>({routing_weights, index_Add_2, 0}, {{"batch_dims", 0}});
    auto index_Reshape_8_2 = makePattern<opset1::Reshape>({index_Gather_5, {0,1}}, {{"special_zero", true}});
    auto mul_Multiply_3 = makePattern<opset1::Multiply>({down_linear_MatMul, index_Gather_5 | index_Reshape_8_2}, {{"auto_broadcast", "numpy"}});
    auto index_add__Broadcast_26 = makePattern<opset3::Broadcast>({mul_Multiply_3, index_add__ShapeOf_22}, {{"mode", "bidirectional"}});
    auto index_add__ScatterElementsUpdate_8 = makePattern<opset12::ScatterElementsUpdate>({final_hidden_states, index_add__Broadcast_25, index_add__Broadcast_26, 0}, {{"reduction", "sum"}, {"use_init_val", true}});

    auto result = index_add__ScatterElementsUpdate_8;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto hidden_size = static_cast<size_t>(validator["hidden_size"]);
        auto expert_no = static_cast<int>(validator["expert_no"]);

        auto expert_mask_node = pattern_map.at(expert_mask);
        auto ps = expert_mask_node.get_partial_shape();
        if (ps.rank().is_dynamic() || ps[0].is_dynamic() || ps[1].is_dynamic()) {
            std::cout << "expert_mask ps is dynamic " << ps << "\n";
            return false;
        }

        auto expert_num = ps[0].get_length();
        auto topk = ps[1].get_length();

        auto last_node = pattern_map.at(index_add__ScatterElementsUpdate_8).get_node_shared_ptr();

        op::internal::MOEExpert::ConstsPerExpert consts;
#define GET_MATMUL_PARAM(mat, idx) \
        if (pattern_map.at(weight_const##idx).get_node()) { \
            mat[0] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weight_const##idx).get_node_shared_ptr());    \
        }   \
        if (pattern_map.at(scale_const##idx).get_node()) { \
            mat[1] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scale_const##idx).get_node_shared_ptr());    \
        }   \
        if (pattern_map.at(zp_const##idx).get_node()) { \
            mat[2] = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zp_const##idx).get_node_shared_ptr());    \
        }

        GET_MATMUL_PARAM(consts.gate, 0)
        GET_MATMUL_PARAM(consts.up, 1)
        GET_MATMUL_PARAM(consts.down, 2)
#undef GET_MATMUL_PARAM

        op::internal::MOEExpert::Config config;
        config.expert_num = expert_num;
        config.hidden_size = hidden_size;
        config.topk = topk;

        OutputVector new_args(4);
        // [final_hidden_states, hidden_states, expert_mask, routing_weights]
        new_args[0] = pattern_map.at(final_hidden_states).get_node_shared_ptr();
        new_args[1] = pattern_map.at(hidden_states).get_node_shared_ptr();
        new_args[2] = pattern_map.at(expert_mask).get_node_shared_ptr();
        new_args[3] = pattern_map.at(routing_weights).get_node_shared_ptr();
        if (new_args[0].get_node_shared_ptr()->get_type_info() == op::internal::MOEExpert::get_type_info_static()) {
            auto moe = ov::as_type_ptr<op::internal::MOEExpert>(new_args[0].get_node_shared_ptr());
            moe->add_consts(expert_no, consts);

            ov::replace_node(last_node, moe);
            register_new_node(moe);
        } else {
            OPENVINO_ASSERT(expert_no == 0, "MOE expert must begin with 0, current: ", expert_no);
            auto new_node =
                std::make_shared<op::internal::MOEExpert>(new_args,
                                                          config,
                                                          std::vector<op::internal::MOEExpert::ConstsPerExpert>{consts});

            new_node->set_friendly_name("moe_expert");

            ov::replace_node(last_node, new_node);
            register_new_node(new_node);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::FuseMoeExpertRouter::FuseMoeExpertRouter() {
    MATCHER_SCOPE(FuseMoeExpertRouter);

    // param1: [batch*seq, 2048]
    auto final_hidden_states = makePattern(ov::Rank(2));
    auto router_logits = makePattern(ov::Rank(2));
    // f32[?,128]
    auto softmax_Softmax = makePattern<opset8::Softmax>({router_logits}, {{"axis", 1}});
    auto topk_TopK = makePattern<opset11::TopK>({softmax_Softmax, pattern::any_input()});
    topk_TopK->set_output_size(2);
    auto sum_ReduceSum = makePattern<opset1::ReduceSum>({topk_TopK->output(0), {-1}}, {{"keep_dims", true}});
    auto div__Divide = makePattern<opset1::Divide>({topk_TopK->output(0), sum_ReduceSum}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto one_hot_OneHot = makePattern<opset1::OneHot>({topk_TopK->output(1), pattern::any_input(), pattern::any_input(), pattern::any_input()}, {{"axis", 2}});
    // param2: expert_mask: [128, 8, batch]
    auto permute_Transpose = makePattern<opset1::Transpose>({one_hot_OneHot, {2, 1, 0}});

    // hidden_states_2d: f32[-1, 2048]
    auto view_Reshape = makePattern(ov::Rank(2));
    // param1: hidden_states: f32[1, -1, 2048]
    auto unsqueeze_Unsqueeze = makePattern<opset1::Unsqueeze>({view_Reshape, 0});

    auto unsqueeze_Unsqueeze_1 = makePattern<opset1::Unsqueeze>({div__Divide, 2});
    auto index_ShapeOf_1 = makePattern<opset3::ShapeOf>({unsqueeze_Unsqueeze_1}, {{"output_type", "i32"}});
    auto index_Slice = makePattern<opset8::Slice>({index_ShapeOf_1, {0}, {2}, {1}, {0}});
    auto index_ReduceProd = makePattern<opset1::ReduceProd>({index_Slice, 0}, {{"keep_dims", true}});
    auto index_Concat = makePattern<opset1::Concat>({index_ReduceProd, {-1}}, {{"axis", 0}});
    // param4: routing weights: [self.topk * batch, 1]
    auto index_Reshape = makePattern<opset1::Reshape>({unsqueeze_Unsqueeze_1, index_Concat}, {{"special_zero", true}});

    auto moe_expert03 = makePattern<ov::op::internal::MOEExpert>({final_hidden_states, unsqueeze_Unsqueeze, permute_Transpose, index_Reshape});
    auto result = moe_expert03;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        // router_logits: i32[batch*seq, 128]
        auto router_logits_node = pattern_map.at(router_logits).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto hidden_states_2d = pattern_map.at(view_Reshape).get_node_shared_ptr();
        // f32[batch*seq, 2048]
        auto moe_node = pattern_map.at(moe_expert03).get_node_shared_ptr();

        auto moe = ov::as_type_ptr<op::internal::MOEExpert>(moe_node);

        OutputVector new_args(2);
        // hidden_states_2d: f32[batch*seq, 2048]
        // router_logits: i32[batch*seq, 128]
        new_args[0] = hidden_states_2d;
        new_args[1] = router_logits_node;
        moe->set_arguments(new_args);
        auto cfg = moe->get_config();
        cfg.fused_router_logic = true;
        moe->set_config(cfg);

        moe->set_friendly_name("moe_expert_router");

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}