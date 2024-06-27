// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector depthwise_conv2d(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = node.get_decoder();
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    auto group = node.get_attribute<int32_t>("group");
    ov::OutputVector inputs = {
        node.get_input(0),
        tensorflow::make_reshape(tensorflow::make_transpose(node.get_input(1), {1, 2, 3, 0}), {0, 0, -1, group})};
    auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs);
    get_conv(output,
             context,
             decoder,
             &ov::frontend::tensorflow::op::translate_depthwise_conv_2d_native_op,
             {0, 1, 2, 3});
    get_bias(output, node, decoder);
    get_activation(output, decoder);
    output[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
