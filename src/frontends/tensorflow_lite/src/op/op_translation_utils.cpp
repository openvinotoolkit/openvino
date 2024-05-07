// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_translation_utils.hpp"

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

void set_output_names(const ov::frontend::tensorflow_lite::NodeContext& node, OutputVector& outputs) {
    const auto& decoder_with_name =
        std::dynamic_pointer_cast<tensorflow_lite::DecoderBaseOperation>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder_with_name != nullptr, "Unexpected decoder during operation translation.");
    FRONT_END_GENERAL_CHECK(outputs.size() == decoder_with_name->get_output_size(),
                            "Unexpected decoder during operation translation.");
    for (size_t i = 0; i < decoder_with_name->get_output_size(); ++i) {
        outputs[i].set_names({decoder_with_name->get_output_tensor_name(i)});
    }
}

void del_output_names(OutputVector& outputs) {
    for (auto& output : outputs) {
        output.set_names({});
    }
}

void get_conv(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&),
              ov::AxisVector transpose_axes) {
    ov::OutputVector inputs = {node.get_input(0),
                               ov::frontend::tensorflow::make_transpose(node.get_input(1), transpose_axes)};
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    output = converter(context);
    del_output_names(output);
}

void get_pool(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&)) {
    ov::OutputVector inputs = {node.get_input(0)};
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    output = converter(context);
    del_output_names(output);
}

void get_bias(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder) {
    if (node.get_input_size() == 3) {
        const OutputVector inputs_for_bias = {output[0], node.get_input(2)};
        auto context_for_bias_add = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs_for_bias);
        output = ov::frontend::tensorflow::op::translate_binary_op<ov::opset10::Add>(context_for_bias_add);
        del_output_names(output);
    }
}

void get_activation(ov::OutputVector& output,
                    const ov::frontend::tensorflow_lite::NodeContext& node,
                    const std::string& activation) {
    auto context = ov::frontend::tensorflow_lite::NodeContext(node.get_decoder(), output);
    if (activation == "RELU") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>(context);
    } else if (activation == "RELU6") {
        output = ov::frontend::tensorflow::op::translate_relu_6_op(context);
    } else if (activation == "TANH") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Tanh>(context);
    } else if (activation == "RELU_N1_TO_1") {
        auto clamp = std::make_shared<opset10::Clamp>(output[0], -1.0f, 1.0f);
        clamp->set_friendly_name(context.get_name());
        output = clamp->outputs();
    } else if (activation == "SIGN_BIT") {
        auto zero = std::make_shared<opset10::ConvertLike>(opset10::Constant::create(element::i32, {}, {0}), output[0]);
        auto less = std::make_shared<opset10::Less>(output[0], zero);
        less->set_friendly_name(context.get_name());
        output = less->outputs();
    } else {
        FRONT_END_GENERAL_CHECK(activation == "NONE",
                                "Unknown Activation fused to ",
                                node.get_decoder()->get_op_type(),
                                ": ",
                                activation);
    }
    del_output_names(output);
}

void get_activation(ov::OutputVector& output,
                    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>& decoder) {
    auto context_for_activation = ov::frontend::tensorflow_lite::NodeContext(decoder, output);
    const auto activation = decoder->get_attribute("activation").as<std::string>();
    get_activation(output, context_for_activation, activation);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
