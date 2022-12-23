// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"
#include "op_translation_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

void set_output_names(const ov::frontend::tensorflow::NodeContext& node, OutputVector& outputs) {
    const auto& decoder_with_name = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder_with_name != nullptr, "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    FRONT_END_GENERAL_CHECK(outputs.size() == decoder_with_name->get_output_size(), "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    for (size_t i = 0; i < decoder_with_name->get_output_size(); ++i) {
        outputs[i].set_names({decoder_with_name->get_output_tensor_name(i)});
    }
}

void del_output_names(OutputVector& outputs) {
    for (auto& output : outputs) {
        output.set_names({});
    }
}


void get_conv(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder, ov::OutputVector(*converter)(const ov::frontend::tensorflow::NodeContext&)) {
    ov::OutputVector inputs = {node.get_input(0),
                               ov::frontend::tensorflow::make_transpose(node.get_input(1), ov::AxisVector{1, 2, 3, 0})};
    auto context = ov::frontend::tensorflow::NodeContext(decoder, inputs);
    output = converter(context);
}


void get_bias(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder) {
    if (node.get_input_size() == 3) {
        const OutputVector inputs_for_bias = {output[0], node.get_input(2)};
        auto context_for_bias_add = ov::frontend::tensorflow::NodeContext(decoder, inputs_for_bias);
        output = ov::frontend::tensorflow::op::translate_binary_op<ov::opset10::Add>(context_for_bias_add);
    }
}

void get_activation(ov::OutputVector& output, const ov::frontend::tensorflow::NodeContext& node, const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder) {
    auto context_for_activation = ov::frontend::tensorflow::NodeContext(decoder, output);
    const auto activation = decoder->get_attribute("activation").as<std::string>();
    if (activation == "RELU") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>(context_for_activation);
    } else if (activation == "RELU6") {
        output = ov::frontend::tensorflow::op::translate_relu_6_op(context_for_activation);
    } else if (activation == "TANH") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Tanh>(context_for_activation);
    } else {
        // TODO: Fused activation to support:
        //          RELU_N1_TO_1 = 2,
        //          SIGN_BIT = 5,
        if (activation != "NONE") {
            FRONT_END_THROW("Unknown Activation in CONV_2D: " + activation);
        }
    }
}


}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov