// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_translation_utils.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector fully_connected(const ov::frontend::tensorflow_lite::NodeContext& node) {
    using FCOptions = tflite::FullyConnectedOptions;
    const auto& decoder = get_decoder(node);
    auto data = node.get_input(0);
    auto weights = node.get_input(1);
    if (decoder->get_attribute(&FCOptions::weights_format) != tflite::FullyConnectedOptionsWeightsFormat_DEFAULT) {
        FRONT_END_NOT_IMPLEMENTED(
            "FullyConnectedOptions::weights_format != FullyConnectedOptionsWeightsFormat_DEFAULT");
    }
    if (!decoder->get_attribute(&FCOptions::keep_num_dims)) {
        auto new_shape = make_shared<opset10::Concat>(
            OutputVector{opset10::Constant::create(element::i64, {1}, {-1}),
                         ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(weights, {1})},
            0);
        data = make_shared<opset10::Reshape>(data, new_shape, false);
    }
    auto output = std::make_shared<opset10::MatMul>(data, weights, false, true)->outputs();
    get_bias(output, node, decoder);
    auto activation_name =
        EnumNameActivationFunctionType(decoder->get_attribute(&FCOptions::fused_activation_function));
    get_activation(output, node, activation_name);
    output[0].get_node_shared_ptr()->set_friendly_name(decoder->get_op_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
