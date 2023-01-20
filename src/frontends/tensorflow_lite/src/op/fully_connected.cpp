// Copyright (C) 2018-2022 Intel Corporation
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

OutputVector fully_connected(const ov::frontend::tensorflow_lite::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto& decoder = std::dynamic_pointer_cast<DecoderFlatBuffer>(node.get_decoder());
    FRONT_END_GENERAL_CHECK(decoder != nullptr,
                            "Unexpected decoder during operation translation. Expected DecoderFlatBuffer");

    auto data = node.get_input(0);
    auto weights = node.get_input(1);
    if (decoder->get_attribute(&tflite::FullyConnectedOptions::weights_format) !=
        tflite::FullyConnectedOptionsWeightsFormat_DEFAULT) {
        FRONT_END_NOT_IMPLEMENTED(
            "FullyConnectedOptions::weights_format != FullyConnectedOptionsWeightsFormat_DEFAULT");
    }
    if (!decoder->get_attribute(&tflite::FullyConnectedOptions::keep_num_dims)) {
        // Everything is 2D now -- insert Reshape
        // weights = Reshape;
    }

    auto output = std::make_shared<opset10::MatMul>(data, weights, false, true)->outputs();
    get_activation(output,
                   node,
                   EnumNameActivationFunctionType(
                       decoder->get_attribute(&tflite::FullyConnectedOptions::fused_activation_function)));
    output[0].get_node_shared_ptr()->set_friendly_name(decoder->get_op_name());
    return output;
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
