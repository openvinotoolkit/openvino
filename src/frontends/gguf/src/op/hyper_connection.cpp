// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ggml hyper-connection operations (Qwen3.8-Flash-Next and DeepSeek-V4 residual streams).
// Shapes are OpenVINO order, the reverse of ggml's ne.

#include <limits>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

using namespace ov::op;

namespace {

std::shared_ptr<v0::Constant> i64s(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}

int64_t static_dim(const ov::Output<ov::Node>& value, int64_t axis, const char* what) {
    const auto& shape = value.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static(), what, " requires a static rank");
    const auto rank = shape.rank().get_length();
    const auto dim = shape[axis < 0 ? rank + axis : axis];
    FRONT_END_OP_CONVERSION_CHECK(dim.is_static(), what, " requires a static dimension ", axis);
    return dim.get_length();
}

ov::Output<ov::Node> f32(const ov::Output<ov::Node>& value) {
    return value.get_element_type() == ov::element::f32
               ? value
               : std::make_shared<v0::Convert>(value, ov::element::f32)->output(0);
}

}  // namespace

// ggml_dsv4_hc_pre: x [1, T, hc, E] collapsed to [1, 1, T, E] by per-stream weights [1, 1, T, hc],
// or (gated) scale * sum_h x * sigmoid(gate) with a gate shaped like x.
OutputVector translate_dsv4_hc_pre(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = f32(context.get_input(0));
    auto weights = f32(context.get_input(1));
    const int64_t streams = static_dim(x, -2, "DSV4_HC_PRE input");
    const bool gated = context.get_attribute<bool>("gated", weights.get_partial_shape() == x.get_partial_shape());
    ov::Output<ov::Node> factors =
        gated ? std::make_shared<v0::Sigmoid>(weights)->output(0)
              : std::make_shared<v1::Reshape>(weights, i64s({0, -1, streams, 1}), true)->output(0);
    ov::Output<ov::Node> summed =
        std::make_shared<v1::ReduceSum>(std::make_shared<v1::Multiply>(x, factors), i64s({2}), false);
    const float scale = context.get_attribute<float>("scale", 1.0f);
    if (scale != 1.0f)
        summed = std::make_shared<v1::Multiply>(summed, v0::Constant::create(ov::element::f32, {}, {scale}));
    return rename_outputs_with_suffix({std::make_shared<v0::Unsqueeze>(summed, i64s({1}))}, context.get_name());
}

// ggml_dsv4_hc_post: out[t, dst] = x[t] * post[t, dst] + sum_src comb[t, src, dst] * residual[t, src]
// with x [1, 1, T, E], residual [1, T, hc, E], post [1, 1, T, hc] and the optional comb
// [1, T, hc_src, hc_dst]; without comb the residual passes through per stream.
OutputVector translate_dsv4_hc_post(const NodeContext& context) {
    num_inputs_check(context, 3, 4);
    auto residual = f32(context.get_input(1));
    const int64_t width = static_dim(residual, -1, "DSV4_HC_POST residual");
    const int64_t streams = static_dim(residual, -2, "DSV4_HC_POST residual");
    auto x = std::make_shared<v1::Reshape>(f32(context.get_input(0)), i64s({0, -1, 1, width}), true);
    auto post = std::make_shared<v1::Reshape>(f32(context.get_input(2)), i64s({0, -1, streams, 1}), true);
    ov::Output<ov::Node> carried = residual;
    if (context.get_input_size() == 4)
        carried = std::make_shared<v0::MatMul>(f32(context.get_input(3)), residual, true, false);
    auto result = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(x, post), carried);
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
