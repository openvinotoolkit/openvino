// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>
#include <utility>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

namespace {
// Resolve the two GLU halves: either two explicit inputs, or one combined tensor split along the
// last axis (floor division; an odd trailing element is dropped). Applies the "swapped" flag.
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> get_glu_inputs(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        auto combined = context.get_input(0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    if (context.get_attribute<bool>("swapped")) {
        std::swap(src0, src1);
    }
    return {src0, src1};
}
}  // namespace

OutputVector translate_glu_swiglu(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);
    auto src0 = inputs.first;
    auto src1 = inputs.second;

    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src0);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src0, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

// gpt-oss gated SiLU: clamp gate to (-inf, limit], scale by alpha, sigmoid-gate, then multiply by
// (clamp(up, -limit, limit) + 1). alpha/limit come from the decoder as typed float attributes.
OutputVector translate_glu_swiglu_oai(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);
    auto src0 = inputs.first;
    auto src1 = inputs.second;

    const float alpha = context.get_attribute<float>("glu_alpha");
    const float limit = context.get_attribute<float>("glu_limit");

    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<float>::infinity(), limit);
    auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {alpha});
    auto scaled_gate = std::make_shared<ov::op::v1::Multiply>(gate, alpha_const);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(scaled_gate);
    auto out_glu = std::make_shared<ov::op::v1::Multiply>(gate, sigmoid);

    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto up_plus_one = std::make_shared<ov::op::v1::Add>(up, one);
    auto res = std::make_shared<ov::op::v1::Multiply>(out_glu, up_plus_one);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
