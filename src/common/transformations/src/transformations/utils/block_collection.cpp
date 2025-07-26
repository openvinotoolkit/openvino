// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/utils/block_collection.hpp"

#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/power.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass::pattern::blocks {

using namespace ov::op;

std::shared_ptr<Node> l2_norm_block(const Output<Node>& input) {
    auto pow = wrap_type<v1::Power>({input, any_input()});
    auto var = wrap_type<v1::ReduceMean, v1::ReduceSum>({pow, any_input()});
    auto sqrt = wrap_type<v0::Sqrt>({var});
    auto div = wrap_type<v1::Divide>({input, sqrt});

    auto scale = wrap_type<v1::Multiply>({div, any_input()});
    auto shift = optional<v1::Add>({scale, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{input}, OutputVector{shift}, "l2_norm");
}

std::shared_ptr<Node> dq_constant_block() {
    auto constant = wrap_type<v0::Constant>();
    auto opt_convert = pattern::optional<v0::Convert>({constant});

    auto zp = any_input();
    auto zp_convert = pattern::optional<v0::Convert>({zp});
    auto zp_shape = pattern::optional<v1::Reshape, v0::Unsqueeze>({zp_convert, any_input()});
    auto subtract = pattern::optional<v1::Subtract>({opt_convert, zp_shape});

    auto scale = any_input();
    auto scale_convert = pattern::optional<v0::Convert>({scale});
    auto scale_shape = pattern::optional<v1::Reshape, v0::Unsqueeze>({scale_convert, any_input()});

    auto mul = wrap_type<v1::Multiply>({subtract, scale_shape});
    auto block = std::make_shared<pattern::op::Block>(OutputVector{}, OutputVector{mul}, "dq_constant");

    REGISTER_ANCHORS(block, constant, zp, scale, mul);
    return block;
}

std::shared_ptr<ov::Node> attention_mask() {
    auto mask = wrap_type<v0::Parameter>();
    auto mask_unsqueeze_1 = wrap_type<v0::Unsqueeze>({mask, any_input()});
    auto mask_unsqueeze_2 = wrap_type<v0::Unsqueeze>({mask_unsqueeze_1, any_input()});
    auto mul = wrap_type<v1::Multiply>({mask_unsqueeze_2, any_input()});
    auto add = wrap_type<v1::Add>({mul, any_input()});
    auto mask_slice = wrap_type<v8::Slice>({add, any_input(), any_input(), any_input(), any_input()});
    return std::make_shared<pattern::op::Block>(OutputVector{}, OutputVector{mask_slice}, "attention_mask");
}

// RoPE? Q, K
std::shared_ptr<Node> sdpa_preprocessing_block(const Output<Node>& input) {
    auto reshape = wrap_type<v1::Reshape>({input, any_input()});
    auto transpose_1 = optional<v1::Transpose>({reshape, any_input()});
    auto var_split = wrap_type<v1::VariadicSplit>({transpose_1, any_input(), any_input()});
    var_split->set_output_size(2);

    auto mul_1 = wrap_type<v1::Multiply>({var_split->output(0), any_input()});
    auto concat = wrap_type<v0::Concat>({mul_1, var_split->output(1)});
    auto mul_2 = wrap_type<v1::Multiply>({concat, any_input()});

    auto mul_3 = wrap_type<v1::Multiply>({reshape, any_input()});
    auto transpose_2 = optional<v1::Transpose>({mul_3, any_input()});
    auto add = wrap_type<v1::Add>({transpose_2, any_input()});  // todo: use mul_2 as 2nd input

    return std::make_shared<pattern::op::Block>(OutputVector{input}, OutputVector{add}, "sdpa_preprocessing");
}

std::shared_ptr<Node> sdpa_block(const Output<Node>& q, const Output<Node>& k, const Output<Node>& v) {
    auto kT = wrap_type<v1::Transpose>({k, any_input()});
    auto scale = optional<v1::Multiply>({kT, any_input()});
    auto qk = wrap_type<v0::MatMul>({q, scale});
    auto bias_add = wrap_type<v1::Add>({qk, any_input()});
    auto softmax = wrap_type<v8::Softmax>({bias_add});
    auto qkv = wrap_type<v0::MatMul>({softmax, v});

    return std::make_shared<pattern::op::Block>(OutputVector{q, k, v}, OutputVector{qkv}, "sdpa");
}

std::shared_ptr<Node> post_sdpa_projection_block(const Output<Node>& qkv) {
    auto t2 = wrap_type<v1::Transpose>({qkv, any_input()});
    auto reshaped = wrap_type<v1::Reshape>({t2, any_input()});
    auto proj = wrap_type<v0::MatMul>({reshaped, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{qkv}, OutputVector{proj}, "post_sdpa_projection_block");
}

}  // namespace ov::pass::pattern::blocks