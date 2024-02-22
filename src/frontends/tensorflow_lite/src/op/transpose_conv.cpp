// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::frontend::tensorflow::op;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector transpose_conv(const ov::frontend::tensorflow_lite::NodeContext& node) {
    using Opts = tflite::TransposeConvOptions;
    const auto& decoder = get_decoder(node);
    const std::map<std::string, ov::Any> attrs{
        {"strides",
         std::vector<int64_t>{1, decoder->get_attribute(&Opts::stride_h), decoder->get_attribute(&Opts::stride_w), 1}},
        {"padding", std::string(EnumNamePadding(decoder->get_attribute(&Opts::padding)))},
        {"data_format", "NHWC"},
        {"dilations", std::vector<int64_t>{1, 1, 1, 1}},
    };
    OutputVector inputs = {
        node.get_input(0),
        ov::frontend::tensorflow::make_transpose(node.get_input(1), ov::AxisVector{1, 2, 0, 3}),
        node.get_input(2),
    };
    auto outputs =
        attribute_helper(node, attrs, translate_conv_2d_backprop_input_op, "Conv2DBackpropInput", true, inputs);
    if (node.get_input_size() == 4) {
        const OutputVector& inputs_for_bias = {outputs[0], node.get_input(3)};
        auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs_for_bias);
        outputs = attribute_helper(node, {}, translate_binary_op<ov::opset10::Add>, "", true, inputs_for_bias);
    }
    outputs[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return outputs;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
