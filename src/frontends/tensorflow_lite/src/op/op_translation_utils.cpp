// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_translation_utils.hpp"

#include <functional>
#include <map>
#include <string>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "openvino_conversions.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

void set_output_names(const ov::frontend::tensorflow_lite::NodeContext& node, OutputVector& outputs) {
    const auto& decoder_with_name = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder_with_name != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    FRONT_END_GENERAL_CHECK(outputs.size() == decoder_with_name->get_output_size(),
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
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
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&)) {
    ov::OutputVector inputs = {node.get_input(0),
                               ov::frontend::tensorflow::make_transpose(node.get_input(1), ov::AxisVector{1, 2, 3, 0})};
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    output = converter(context);
}

void get_pool(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder,
              ov::OutputVector (*converter)(const ov::frontend::NodeContext&)) {
    ov::OutputVector inputs = {node.get_input(0)};
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    output = converter(context);
}

void get_bias(ov::OutputVector& output,
              const ov::frontend::NodeContext& node,
              const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder) {
    if (node.get_input_size() == 3) {
        const OutputVector inputs_for_bias = {output[0], node.get_input(2)};
        auto context_for_bias_add = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs_for_bias);
        output = ov::frontend::tensorflow::op::translate_binary_op<ov::opset10::Add>(context_for_bias_add);
    }
}

void get_activation(ov::OutputVector& output,
                    const ov::frontend::tensorflow_lite::NodeContext& node,
                    const std::string& activation) {
    if (activation == "RELU") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>(node);
    } else if (activation == "RELU6") {
        output = ov::frontend::tensorflow::op::translate_relu_6_op(node);
    } else if (activation == "TANH") {
        output = ov::frontend::tensorflow::op::translate_unary_op<opset10::Tanh>(node);
    } else {
        // TODO: Fused activation to support:
        //          RELU_N1_TO_1 = 2,
        //          SIGN_BIT = 5,
        if (activation != "NONE") {
            FRONT_END_THROW("Unknown Activation fused to " + node.get_decoder()->get_op_type() + ": " + activation);
        }
    }
}

void get_activation(ov::OutputVector& output,
                    const std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap>& decoder) {
    auto context_for_activation = ov::frontend::tensorflow_lite::NodeContext(decoder, output);
    const auto activation = decoder->get_attribute("activation").as<std::string>();
    get_activation(output, context_for_activation, activation);
}

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderMap> get_pool_decoder_map(
    const std::string& new_type_name,
    const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");

    const std::map<std::string, ov::Any> attrs{
        {"strides",
         std::vector<int64_t>{1,
                              decoder->get_attribute(&tflite::Pool2DOptions::stride_h),
                              decoder->get_attribute(&tflite::Pool2DOptions::stride_w),
                              1}},
        {"padding", std::string(EnumNamePadding(decoder->get_attribute(&tflite::Pool2DOptions::padding)))},
        {"ksize",
         std::vector<int64_t>{1,
                              decoder->get_attribute(&tflite::Pool2DOptions::filter_height),
                              decoder->get_attribute(&tflite::Pool2DOptions::filter_width),
                              1}},
        {"data_format", "NHWC"},
        {"activation",
         EnumNameActivationFunctionType(decoder->get_attribute(&tflite::Pool2DOptions::fused_activation_function))},
    };
    return std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(node.get_decoder(), attrs, new_type_name, true);
}

OutputVector attribute_helper(const ov::frontend::tensorflow_lite::NodeContext& node,
                              const std::map<std::string, ov::Any>& attrs,
                              ov::OutputVector (*converter)(const ov::frontend::NodeContext&),
                              std::string new_op_type,
                              bool empty_name) {
    const auto& original_decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(original_decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");
    auto decoder = std::make_shared<ov::frontend::tensorflow_lite::DecoderMap>(
        original_decoder,
        attrs,
        (new_op_type.empty() ? original_decoder->get_op_type() : new_op_type),
        empty_name);

    OutputVector inputs(node.get_input_size());
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        inputs[i] = node.get_input(static_cast<int>(i));
    }
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    return converter(context);
}
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
