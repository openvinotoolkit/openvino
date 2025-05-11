// Copyright (C) 2018-2025 Intel Corporation
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
    const auto& decoder = node.get_decoder();
    OutputVector inputs = {
        node.get_input(0),
        ov::frontend::tensorflow::make_transpose(node.get_input(1), ov::AxisVector{1, 2, 0, 3}),
        node.get_input(2),
    };
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    auto outputs = translate_conv_2d_backprop_input_op(context);
    del_output_names(outputs);
    if (node.get_input_size() == 4) {
        const OutputVector& inputs_for_bias = {outputs[0], node.get_input(3)};
        auto context_bias = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs_for_bias);
        auto outputs = translate_binary_op<ov::opset10::Add>(context_bias);
        del_output_names(outputs);
    }
    outputs[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return outputs;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
