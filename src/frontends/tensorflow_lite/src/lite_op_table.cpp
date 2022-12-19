// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lite_op_table.hpp"
#include "decoder_map.hpp"

#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector conv2d(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto* conv_opts = node.get_decoder()->get_attribute("Conv2DOptions").as<const tflite::Conv2DOptions*>();
    const map<string, ov::Any> attrs {
            {"strides", vector<int64_t>{1, conv_opts->stride_h(), conv_opts->stride_w(), 1}},
            {"padding", string(EnumNamePadding(conv_opts->padding()))},
            {"dilations", vector<int64_t>{1, conv_opts->dilation_h_factor(), conv_opts->dilation_w_factor(), 1}},
            {"data_format", "NHWC"},
    };
    auto decoder_for_tf_translator = std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(node.get_decoder(), attrs, "Conv2D", true);
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2, "Unexpected number of input in node of type=", node.get_op_type(), " name=", node.get_name());

    // Convolution
    ov::OutputVector inputs = {node.get_input(0),
                               ov::frontend::tensorflow::make_transpose(node.get_input(1), ov::AxisVector{1, 2, 3, 0})};
    const auto context_for_tf = ov::frontend::tensorflow::NodeContext(decoder_for_tf_translator, inputs);
    auto output = ov::frontend::tensorflow::op::translate_conv_2d_op(context_for_tf);
    // Bias
    if (node.get_input_size() == 3) {
        const OutputVector inputs_for_bias = {output[0], node.get_input(2)};
        auto context_for_bias_add = ov::frontend::tensorflow::NodeContext(decoder_for_tf_translator, inputs_for_bias);
        output = ov::frontend::tensorflow::op::translate_binary_op<opset10::Add>(context_for_bias_add);
    }
    // Activation
    auto context_for_activation = ov::frontend::tensorflow::NodeContext(decoder_for_tf_translator, output);
    const auto activation = conv_opts->fused_activation_function();
    if (activation == tflite::ActivationFunctionType_RELU) {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>(context_for_activation);
    } else if (activation == tflite::ActivationFunctionType_RELU6) {
        output = ov::frontend::tensorflow::op::translate_relu_6_op(context_for_activation);
    } else if (activation == tflite::ActivationFunctionType_TANH) {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset8::Tanh>(context_for_activation);
    } else {
        // TODO: Fused activation to support:
        //          RELU_N1_TO_1 = 2,
        //          SIGN_BIT = 5,
        if (activation != tflite::ActivationFunctionType_NONE) {
            FRONT_END_THROW("Unknown Activation in CONV_2D: " + string(tflite::EnumNameActivationFunctionType(activation)));
        }
    }


    // tensor translation:
        // quantization
        // sparsity
        // etc.
        //    VT_SHAPE = 4,                 +
        //    VT_TYPE = 6,                  +
        //    VT_BUFFER = 8,                +
        //    VT_NAME = 10,                 TODO
        //    VT_QUANTIZATION = 12,         TODO
        //    VT_IS_VARIABLE = 14,          TODO
        //    VT_SPARSITY = 16,             TODO
        //    VT_SHAPE_SIGNATURE = 18,      ????
        //    VT_HAS_RANK = 20,             ????
        //    VT_VARIANT_TENSORS = 22       ????
    // set node names, and output names
    // node name is empty
    // tensor name is set by Tensor.get_name()
    const auto& decoder = dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr, "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    output[0].set_names({decoder->get_output_tensor_name(0)});
    return output;
}

std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"CONV_2D", conv2d},
    };
};

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov