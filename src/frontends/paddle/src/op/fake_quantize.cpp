// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "ngraph_ops/channel_fake_quant_internal.hpp"
#include "ngraph_ops/fake_dequant_internal.hpp"
#include "ngraph_ops/fake_quant_dequant_internal.hpp"
#include "ngraph_ops/fake_quant_internal.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

// about quant in paddle: https://github.com/PaddlePaddle/PaddleSlim/issues/937

NamedOutputs fake_quantize_range_abs_max(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto scale = node.get_input("InScale");
    const auto bit_length = node.get_attribute<int32_t>("bit_length");

    auto fake =
        std::make_shared<ngraph::op::internal::FakeQuantInternal>(x, scale, "fake_quantize_range_abs_max", bit_length);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_quantize_moving_average_abs_max(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto scale = node.get_input("InScale");
    const auto bit_length = node.get_attribute<int32_t>("bit_length");

    auto fake = std::make_shared<ngraph::op::internal::FakeQuantInternal>(x,
                                                                          scale,
                                                                          "fake_quantize_moving_average_abs_max",
                                                                          bit_length);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_quantize_dequantize_moving_average_abs_max(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto scale = node.get_input("InScale");
    const auto bit_length = node.get_attribute<int32_t>("bit_length");

    auto fake = std::make_shared<ngraph::op::internal::FakeQuantDequantInternal>(
        x,
        scale,
        "fake_quantize_dequantize_moving_average_abs_max",
        0,
        bit_length);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_channel_wise_dequantize_max_abs(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto scales = node.get_ng_inputs("Scales");
    PADDLE_OP_CHECK(node, scales.size() <= 2, "scales size should be 1 or 2");
    const auto quant_bits = node.get_attribute<std::vector<int32_t>>("quant_bits");
    const auto quant_axis = node.get_attribute<int32_t>("quant_axis", 0);

    // TODO: why use quant_bits[0], ref:
    // https://github.com/PaddlePaddle/Paddle-Lite/blob/9652eab07f769bfa0ce10f9fe8288fff6f5b0af6/lite/core/optimizer/mir/fusion/quant_dequant_op_fuser.cc#L365
    auto fake =
        std::make_shared<ngraph::op::internal::ChannelFakeQuantInternal>(x,
                                                                         scales[0],
                                                                         scales.size() > 1 ? scales[1] : scales[0],
                                                                         "fake_channel_wise_dequantize_max_abs",
                                                                         quant_axis,
                                                                         quant_bits[0]);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_dequantize_max_abs(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale");
    const auto max_range = node.get_attribute<float>("max_range");

    // find previous quantize node
    std::shared_ptr<ngraph::Node> parent = x.get_node_shared_ptr()->input_value(0).get_node_shared_ptr();
    int bit_length = -1;
    while (parent) {
        const auto& fake_quant_op = std::dynamic_pointer_cast<const ngraph::op::internal::FakeQuantInternal>(parent);
        if (fake_quant_op) {
            bit_length = fake_quant_op->m_bit_length;
            break;
        }
        parent = parent->input_value(0).get_node_shared_ptr();
    }
    PADDLE_OP_CHECK(node, bit_length > 0, "could not find a fake_quantize op");
    auto fake = std::make_shared<ngraph::op::internal::FakeDequantInternal>(x, scale, bit_length, max_range);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_quantize_dequantize_abs_max(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto bit_length = node.get_attribute<int32_t>("bit_length");

    auto fake = std::make_shared<ngraph::op::internal::FakeQuantDequantInternal>(x,
                                                                                 x,
                                                                                 "fake_quantize_dequantize_abs_max",
                                                                                 0,
                                                                                 bit_length);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}

NamedOutputs fake_channel_wise_quantize_dequantize_abs_max(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto bit_length = node.get_attribute<int32_t>("bit_length");
    // no InScale and should get scale from OutScale
    const auto scale = node.get_input("OutScale");
    const auto quant_axis = node.get_attribute<int32_t>("quant_axis", 0);

    auto fake = std::make_shared<ngraph::op::internal::FakeQuantDequantInternal>(
        x,
        scale,
        "fake_channel_wise_quantize_dequantize_abs_max",
        quant_axis,
        bit_length);
    NamedOutputs named_outputs;
    named_outputs["Out"] = {fake->output(0)};

    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
