// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_mask_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "utils/pattern_node.hpp"

#define CALLBACK_LOG(m) std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;

ov::intel_cpu::CausalMaskFusion::CausalMaskFusion() {
    MATCHER_SCOPE(CausalMaskFusion);

    auto query = GenPattern(ov::Rank(4));      // query "f32[1,8,?,64]"
    auto key = GenPattern(ov::Rank(4));        // key "f32[1,8,?,64]"
    auto value = GenPattern(ov::Rank(4));      // value "f32[1,8,?,64]"
    auto attn_mask = GenPattern(ov::Rank(4));  // "f32[1,1,1,?]"

    auto const_4 = GenConst_tril<uint8_t>("u8[1,1,?,?]");

    auto ShapeOf_55964 = GenPattern<opset1::ShapeOf>({key}, "i32[4]");
    auto size_Gather_422 = GenPattern<opset8::Gather>({ShapeOf_55964, {2}, {0}}, nullptr, {{"batch_dims", 0}});
    auto ShapeOf_55965 = GenPattern<opset1::ShapeOf>({query}, "i32[4]");
    auto Multiply_62318 = GenPattern<opset1::Multiply>({ShapeOf_55965, {-1}}, "i32[4]", {{"auto_broadcast", "numpy"}});
    auto size_Gather_419 = GenPattern<opset8::Gather>({Multiply_62318, {2}, {0}}, nullptr, {{"batch_dims", 0}});
    auto sub_Subtract =
        GenPattern<opset1::Add>({size_Gather_422, size_Gather_419}, nullptr, {{"auto_broadcast", "numpy"}});
    auto slice_Unsqueeze_426 = GenPattern<opset1::Reshape>({sub_Subtract, {1}}, nullptr, {{"special_zero", 0}});
    auto ScatterUpdate_59279 = GenPattern<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_426, {0}}, "i32[3]");
    auto slice_Unsqueeze_427 = GenPattern<opset1::Reshape>({size_Gather_422, {1}}, nullptr, {{"special_zero", 0}});
    auto ScatterUpdate_59281 = GenPattern<opset3::ScatterUpdate>({{0, 0, 0}, {2}, slice_Unsqueeze_427, {0}}, "i32[3]");
    auto slice_Slice_429 =
        GenPattern<opset1::StridedSlice>({const_4, ScatterUpdate_59279, ScatterUpdate_59281, {1, 1, 1}},
                                         nullptr,  // "u8[1,1,..2048,2048]",
                                         {{"begin_mask", {1, 1, 0}},
                                          {"end_mask", {1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto ScatterUpdate_59428 =
        GenPattern<opset3::ScatterUpdate>({{0, 0, 0, 0}, {3}, slice_Unsqueeze_427, {0}}, "i32[4]");
    auto slice_Slice_435 =
        GenPattern<opset1::StridedSlice>({slice_Slice_429, {0, 0, 0, 0}, ScatterUpdate_59428, {1, 1, 1, 1}},
                                         nullptr,  // "u8[1,1,..2048,..2048]",
                                         {{"begin_mask", {1, 1, 1, 0}},
                                          {"end_mask", {1, 1, 1, 0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
    auto where_Select = GenPattern<opset1::Select>({slice_Slice_435, {0.0f}, {-FLT_MAX}},
                                                   nullptr,
                                                   {{"auto_broadcast", "numpy"}});  // "f32[1,1,..2048,..2048]"
    auto Add_65111 = GenPattern<opset1::Add>({where_Select, attn_mask}, nullptr, {{"auto_broadcast", "numpy"}});
    auto ShapeOf_56185 = GenPattern<opset1::ShapeOf>({Add_65111}, "i32[4]");
    auto Maximum_65113 =
        GenPattern<opset1::Maximum>({ShapeOf_56185, {1, 1, 1, 1}}, "i32[4]", {{"auto_broadcast", "numpy"}});
    auto Constant_51670 = GenConst({0}, "u8[]");
    auto add_Add_446 = GenPattern<opset1::Broadcast>({Add_65111, Maximum_65113, Constant_51670},
                                                     nullptr,
                                                     {{"mode", "numpy"}});  // "f32[1,1,1..2048,?]"
    auto scaled_dot_product_attention =
        GenPattern<opset12::ScaledDotProductAttention>({query, key, value, add_Add_446},
                                                       nullptr,
                                                       {{"causal", 0}});  // "f32[1,8,?,64]"

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto q = pattern_map.at(query);
        auto k = pattern_map.at(key);
        auto v = pattern_map.at(value);
        auto attn = pattern_map.at(attn_mask);

        auto old_node = root;
        auto new_node = std::make_shared<opset12::ScaledDotProductAttention>(q, k, v, true, attn);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);

        // this new node may match following additional matchers
        // register_new_node(new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(scaled_dot_product_attention, matcher_name);
    this->register_matcher(m, callback);
}
