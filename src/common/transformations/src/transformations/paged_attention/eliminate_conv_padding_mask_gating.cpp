// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/eliminate_conv_padding_mask_gating.hpp"

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace ov::pass {

EliminateConvPaddingMaskGating::EliminateConvPaddingMaskGating() {
    MATCHER_SCOPE(EliminateConvPaddingMaskGating);

    // Pattern: attention_mask -> Slice -> Unsqueeze -> [Convert] -> Multiply -> Add -> Multiply(H, mask_expr)
    auto attn_mask = wrap_type<v0::Parameter>([](const ov::Output<ov::Node>& output) {
        return output.get_names().count("attention_mask");
    });
    auto slice = wrap_type<v8::Slice>({attn_mask, any_input(), any_input(), any_input(), any_input()});
    auto unsqueeze = pattern::optional<v0::Unsqueeze>({slice, any_input()});
    auto convert = pattern::optional<v0::Convert>({unsqueeze});
    auto mul_mask = wrap_type<v1::Multiply>({convert, any_input()});
    auto add = wrap_type<v1::Add>({mul_mask, any_input()});
    auto hidden_states = any_input();
    auto mul_gate = wrap_type<v1::Multiply>({hidden_states, add});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        pm.at(mul_gate).get_node_shared_ptr()->output(0).replace(pm.at(hidden_states));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_gate, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
