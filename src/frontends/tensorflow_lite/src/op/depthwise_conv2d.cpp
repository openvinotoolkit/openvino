// Copyright (C) 2018-2023 Intel Corporation
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
    auto decoder_ = get_decoder(node);
    auto group = decoder_->get_attribute(&tflite::DepthwiseConv2DOptions::depth_multiplier);
    auto decoder = get_conv_decoder_map<tflite::DepthwiseConv2DOptions>("DepthwiseConv2dNative", node);
    FRONT_END_GENERAL_CHECK(node.get_input_size() >= 2,
                            "Unexpected number of input in node of type=",
                            node.get_op_type(),
                            " name=",
                            node.get_name());
    OutputVector output;
    if (group == 1)
        get_conv(output, node, decoder, &ov::frontend::tensorflow::op::translate_conv_2d_op);
    else
        get_conv(output, node, decoder, &ov::frontend::tensorflow::op::translate_depthwise_conv_2d_native_op, {1, 2, 0, 3});
    get_bias(output, node, decoder);
    get_activation(output, decoder);
    output[0].get_node_shared_ptr()->set_friendly_name(node.get_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
