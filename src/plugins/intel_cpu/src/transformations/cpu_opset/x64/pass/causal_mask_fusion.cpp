// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_mask_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "utils/gen_pattern.hpp"
#include "transformations/cpu_opset/x64/op/sdp.hpp"

#define CALLBACK_LOG(m) \
    if (0)              \
        std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;

ov::intel_cpu::CausalMaskFusion::CausalMaskFusion() {
    MATCHER_SCOPE(CausalMaskFusion);

    auto query = GenPattern(ov::Rank(4));      // query "f32[1,8,?,64]"
    auto key = GenPattern(ov::Rank(4));        // key "f32[1,8,?,64]"
    auto value = GenPattern(ov::Rank(4));      // value "f32[1,8,?,64]"
    auto attn_mask = GenPattern(ov::Rank(4));  // "f32[1,1,1,?]"

    auto const_4 = GenConst_tril<uint8_t>("boolean[1,1,?,?]");

    auto size_ShapeOf_11307 = GenPattern<opset3::ShapeOf>({key}, "i32[4]", {{"output_type", "i32"}});
    auto size_Gather_11309 =
        GenPattern<opset8::Gather>({size_ShapeOf_11307, {-2}, {0}}, nullptr, {{"batch_dims", 0}});  // "i32[]"
    auto size_ShapeOf_11300 = GenPattern<opset3::ShapeOf>({query}, "i32[4]", {{"output_type", "i32"}});
    auto size_Gather_11306 =
        GenPattern<opset8::Gather>({size_ShapeOf_11300, {-2}, {0}}, nullptr, {{"batch_dims", 0}});  // "i32[]"
    auto sub_Subtract = GenPattern<opset1::Subtract>({size_Gather_11309, size_Gather_11306},
                                                     nullptr,
                                                     {{"auto_broadcast", "numpy"}});  // "i32[]"
    auto slice_Unsqueeze_11312 = GenPattern<opset1::Unsqueeze>({sub_Subtract, {0}});  // "i32[1]"

    auto ScatterUpdate_254402 =
        GenPattern<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_11312 | sub_Subtract, {0}}, "i32[3]");
    auto slice_Unsqueeze_11313 = GenPattern<opset1::Unsqueeze>({size_Gather_11309, {0}}, "i32[1]");
    auto ScatterUpdate_254404 =
        GenPattern<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_11313 | size_Gather_11309, {0}}, "i32[3]");
    auto slice_Slice_11315 =
        GenPattern<opset1::StridedSlice>({const_4, ScatterUpdate_254402, ScatterUpdate_254404, {1, 1, 1}},
                                         nullptr,
                                         {{"begin_mask", {1, 1, 0}},
                                          {"end_mask", {1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});  // "boolean[1,1,..2048,2048]"
    auto ScatterUpdate_254487 =
        GenPattern<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_11313, {0}}, "i32[4]");
    auto slice_Slice_11321 =
        GenPattern<opset1::StridedSlice>({slice_Slice_11315, {0, 0, 0, 0}, ScatterUpdate_254487, {1, 1, 1, 1}},
                                         nullptr,
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});  // "boolean[1,1,..2048,..2048]"
    auto where_Select = GenPattern<opset1::Select>({slice_Slice_11321, {0.0f}, {-FLT_MAX}},
                                                   nullptr,
                                                   {{"auto_broadcast", "numpy"}});  // "f32[1,1,..2048,..2048]"
    auto Add_263280 = GenPattern<opset1::Add>({where_Select, attn_mask},
                                              nullptr,
                                              {{"auto_broadcast", "numpy"}});  // "f32[?,1,..2048,?]"
    auto ShapeOf_263281 = GenPattern<opset3::ShapeOf>({Add_263280}, "i32[4]", {{"output_type", "i32"}});
    auto size_Gather_11302 = GenPattern<opset8::Gather>({size_ShapeOf_11300, {0}, {0}}, "i32[1]", {{"batch_dims", 0}});
    auto ListConstruct_11326_Concat =
        GenPattern<opset1::Concat>({size_Gather_11302, {-1}, {-1}, {-1}}, "i32[4]", {{"axis", 0}});
    auto expand_Equal = GenPattern<opset1::Equal>({ListConstruct_11326_Concat, {-1, -1, -1, -1}},
                                                  "boolean[4]",
                                                  {{"auto_broadcast", "numpy"}});
    auto expand_Select = GenPattern<opset1::Select>({expand_Equal, {1, 1, 1, 1}, ListConstruct_11326_Concat},
                                                    "i32[4]",
                                                    {{"auto_broadcast", "numpy"}});
    auto Maximum_263282 =
        GenPattern<opset1::Maximum>({ShapeOf_263281, expand_Select}, "i32[4]", {{"auto_broadcast", "numpy"}});
    auto add_Add_11332 = GenPattern<opset3::Broadcast>({Add_263280, Maximum_263282},
                                                       nullptr,
                                                       {{"mode", "numpy"}});  // "f32[?,1,1..2048,?]"
    auto sdpa = GenPattern<opset13::ScaledDotProductAttention>({query, key, value, add_Add_11332},
                                                               nullptr,
                                                               {{"causal", 0}});  // "f32[?,16,?,256]"

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto q = pattern_map.at(query);
        auto k = pattern_map.at(key);
        auto v = pattern_map.at(value);
        auto attn = pattern_map.at(attn_mask);

        auto old_node = root;
        ov::intel_cpu::ScaledDotProductAttentionNode::Config config;
        config.fuse_causal_attn = true;
        auto new_node = std::make_shared<ScaledDotProductAttentionNode>(OutputVector{q, k, v, attn}, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);

        std::cout << "CausalMaskFusion for " << old_node->get_friendly_name() << std::endl;
        // this new node may match following additional matchers
        // register_new_node(new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sdpa, matcher_name);
    this->register_matcher(m, callback);
}