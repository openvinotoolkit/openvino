// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/StateManagementPattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/cc/pass/itt.hpp"

ov::pass::StateManagementPattern::StateManagementPattern() {
    MATCHER_SCOPE(StateManagementPattern);

    auto k_past_var = pattern::wrap_type<op::v3::ReadValue>({pattern::any_input()});
    auto k_past_par = pattern::wrap_type<op::v3::ReadValue>();
    auto k_past = std::make_shared<pattern::op::Or>(OutputVector{pattern::wrap_type<op::v1::Gather>({k_past_par, pattern::any_input(), pattern::any_input()}), k_past_par});
    k_past = std::make_shared<pattern::op::Or>(OutputVector{k_past, pattern::wrap_type<op::v1::Transpose>({k_past, pattern::any_input()})}); //Transpose is used when kv-cache is stored in a not usual layout, example: bloom
    auto k_current = pattern::any_input();
    auto k_current2 = pattern::any_input();
    auto k_current_reshaped = pattern::wrap_type<op::v1::Reshape>({k_current2, pattern::any_input()});
    auto k_concat = pattern::wrap_type<op::v0::Concat>({k_past, std::make_shared<pattern::op::Or>(OutputVector{k_current_reshaped, k_current})});

    auto kv_shaping = [OV_CAPTURE_CPY_AND_THIS](std::shared_ptr<Node> kv_concat) {
        auto interim = pattern::wrap_type<op::v1::StridedSlice>({kv_concat, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<op::v1::StridedSlice>({interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        auto unsqueeze = pattern::wrap_type<op::v0::Unsqueeze>({std::make_shared<pattern::op::Or>(OutputVector{kv_concat, interim}), pattern::any_input()});
        interim = pattern::wrap_type<op::v1::StridedSlice>({unsqueeze, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<op::v1::StridedSlice>({interim, pattern::any_input(), pattern::any_input(), pattern::any_input()});
        interim = pattern::wrap_type<op::v1::Broadcast>({std::make_shared<pattern::op::Or>(OutputVector{unsqueeze, interim}), pattern::any_input()});
        interim = pattern::wrap_type<op::v1::Reshape>({interim, pattern::any_input()});
        return interim;
    };

    auto v_past_var = pattern::wrap_type<op::v3::ReadValue>({pattern::any_input()});
    auto v_past_par = pattern::wrap_type<op::v0::Parameter>();
    auto v_past = std::make_shared<pattern::op::Or>(OutputVector{pattern::wrap_type<op::v1::Gather>({v_past_par, pattern::any_input(), pattern::any_input()}), v_past_par});
    v_past = std::make_shared<pattern::op::Or>(OutputVector{v_past, pattern::wrap_type<op::v1::Transpose>({v_past, pattern::any_input()})});
    auto v_current = pattern::any_input();
    auto v_current2 = pattern::any_input();
    auto v_current_reshaped = pattern::wrap_type<op::v1::Reshape>({v_current2, pattern::any_input()});
    auto v_concat = pattern::wrap_type<op::v0::Concat>({v_past, std::make_shared<pattern::op::Or>(OutputVector{v_current_reshaped, v_current})});

    auto k_shaped = kv_shaping(k_concat);
    auto v_shaped = kv_shaping(v_concat);

    auto k_simply_shaped = pattern::wrap_type<op::v1::Reshape>({k_concat, pattern::any_input()});
    auto v_simply_shaped = pattern::wrap_type<op::v1::Reshape>({v_concat, pattern::any_input()});

    auto k_order = pattern::any_input();
    auto v_order = pattern::any_input();
    
    // KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
    auto k_shaped_transposed = pattern::wrap_type<op::v1::Transpose>({std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped}), k_order});
    auto v_shaped_transposed = pattern::wrap_type<op::v1::Transpose>({std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped}), v_order});

    // Optional pattern to capture alibi slopes (based on pattern from bloom)
    auto alibi = pattern::any_input();
    auto sdpa_mask = pattern::wrap_type<op::v1::Multiply>({pattern::any_input(), alibi});
    sdpa_mask = pattern::wrap_type<op::v1::Reshape>({sdpa_mask, pattern::any_input()});
    sdpa_mask = pattern::wrap_type<op::v1::Reshape>({sdpa_mask, pattern::any_input()});
    sdpa_mask = pattern::wrap_type<op::v1::Select>({pattern::any_input(), pattern::any_input(), sdpa_mask});

    auto q = pattern::any_input();
    auto sdpa = pattern::wrap_type<op::v13::ScaledDotProductAttention>({
        q,
        std::make_shared<pattern::op::Or>(OutputVector{k_concat, k_shaped, k_shaped_transposed, k_simply_shaped}),
        std::make_shared<pattern::op::Or>(OutputVector{v_concat, v_shaped, v_shaped_transposed, v_simply_shaped}),
        std::make_shared<pattern::op::Or>(OutputVector{sdpa_mask, pattern::any_input()})
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        std::cout << "____" << matcher_name << "___Matched___" << std::endl;
        OPENVINO_ASSERT(m.get_pattern_value_map().find(sdpa) != m.get_pattern_value_map().end());
        const auto& pattern_map = m.get_pattern_map();
        OPENVINO_ASSERT(pattern_map.find(sdpa) != pattern_map.end());
        // Why do have double check here?

        // takes option that has 4D instead of fine-grained Reshape analysis
        // it avoids complication in the pattern, but we don't really have many options
        auto take_4d = [OV_CAPTURE_CPY_AND_THIS](std::shared_ptr<Node> option1, std::shared_ptr<Node> option2, std::shared_ptr<Node> option3) {
            // Question: should it be get_partial_shape() of output or input ???
            if (pattern_map.find(option1) != pattern_map.end() && pattern_map.at(option1)->input(0).get_partial_shape().rank().get_length() == 4) {
                return pattern_map.at(option1);
            } else if (pattern_map.at(option2)->input(0).get_partial_shape().rank().get_length() == 4) {
                return pattern_map.at(option2);
            } else {
                return pattern_map.at(option3);
            }
        };

        auto real_k = take_4d(k_current, k_current_reshaped, k_current2);
        auto real_v = take_4d(v_current, v_current_reshaped, v_current2);

        // is_cpu() check required here.
        // kv_cache_type required here (CURRENTLY USING STUB)
        auto kv_cache_type = element::i64; auto k_param = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, PartialShape{-1, -1, -1, -1});
        auto v_parameter = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, PartialShape{-1, -1, -1, -1});
        

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(k_concat, matcher_name);
    register_matcher(m, callback);
}