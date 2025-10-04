// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_sdpa_inputs.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::op;
ov::pass::NormalizeSDPAInputs::NormalizeSDPAInputs() {
    MATCHER_SCOPE(NormalizeSDPAInputs);
    auto query = pattern::any_input(pattern::rank_equals(3));
    auto key = pattern::any_input(pattern::rank_equals(3));
    auto value = pattern::any_input(pattern::rank_equals(3));
    auto trans_q = pattern::wrap_type<op::v1::Transpose>({query, {1, 0, 2}});
    auto trans_k = pattern::wrap_type<op::v1::Transpose>({key, {1, 0, 2}});
    auto trans_v = pattern::wrap_type<op::v1::Transpose>({value, {1, 0, 2}});
    auto attn_mask = pattern::any_input(pattern::rank_equals(3));
    auto sdp0 = wrap_type<ov::op::v13::ScaledDotProductAttention>({trans_q, trans_k, trans_v, attn_mask});
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        std::cout << "matched NormalizeSDPAInputs!!!!!!!" << std::endl;
        const auto& pattern_to_output = m.get_pattern_value_map();
        // auto query_ouput = pattern_to_output.at(query);
        auto const_zero = op::v0::Constant::create(ov::element::i32, {}, std::vector<int32_t>{0});
        auto query_unsuqeeze = std::make_shared<op::v0::Unsqueeze>(pattern_to_output.at(query), const_zero->output(0));
        auto key_unsuqeeze = std::make_shared<op::v0::Unsqueeze>(pattern_to_output.at(key), const_zero->output(0));
        auto val_unsuqeeze = std::make_shared<op::v0::Unsqueeze>(pattern_to_output.at(value), const_zero->output(0));
        auto attn_unsuqeeze = std::make_shared<op::v0::Unsqueeze>(pattern_to_output.at(attn_mask), const_zero->output(0));
        auto const_transpose_order = op::v0::Constant::create(ov::element::i32, {4}, std::vector<int32_t>{0, 2, 1, 3});
        auto trans_q = std::make_shared<op::v1::Transpose>(query_unsuqeeze->output(0), const_transpose_order->output(0));
        auto trans_k = std::make_shared<op::v1::Transpose>(key_unsuqeeze->output(0), const_transpose_order->output(0));
        auto trans_v = std::make_shared<op::v1::Transpose>(val_unsuqeeze->output(0), const_transpose_order->output(0));
        auto new_sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(trans_q, trans_k, trans_v, attn_unsuqeeze, false);
        ov::copy_runtime_info(as_node_vector({pattern_to_output.at(query), pattern_to_output.at(key), pattern_to_output.at(value), pattern_to_output.at(sdp0)}), new_sdpa);
        auto output_unsuqeeze = std::make_shared<op::v0::Squeeze>(new_sdpa, const_zero->output(0));
        ov::replace_node(pattern_to_output.at(sdp0).get_node_shared_ptr(), output_unsuqeeze);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp0, matcher_name);
    register_matcher(m, callback);
}
