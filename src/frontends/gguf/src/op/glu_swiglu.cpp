// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <memory>
#include <utility>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_glu_swiglu(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);
    auto src0 = inputs.first;
    auto src1 = inputs.second;

    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src0);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src0, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
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

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

OutputVector translate_glu_swiglu_clamp(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);
    auto src0 = inputs.first;
    auto src1 = inputs.second;

    const float limit = context.get_attribute<float>("glu_limit");
    auto gate = std::make_shared<ov::op::v0::Clamp>(src0, -std::numeric_limits<float>::infinity(), limit);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(gate);
    auto silu = std::make_shared<ov::op::v1::Multiply>(gate, sigmoid);
    auto up = std::make_shared<ov::op::v0::Clamp>(src1, -limit, limit);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, up);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
