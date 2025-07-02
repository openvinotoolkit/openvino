// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_translation_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector embedding_lookup(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto& decoder = std::dynamic_pointer_cast<DecoderBaseOperation>(node.get_decoder());
    //NOTE: Node Indices for input and input_indices are reversed in TfLite Dialect for
    //Embedding Lookup op as compared to Gather op. Translate function for Gather op cannot
    //be re-used for this reason.
    auto input = node.get_input(1);
    auto input_indices = node.get_input(0);
    int32_t axis_value = 0;
    auto axis = opset10::Constant::create(element::i32, {}, {axis_value});
    auto gather = make_shared<opset10::Gather>(input, input_indices, axis);
    gather->set_friendly_name(node.get_name() + "_gather");
    auto res = make_shared<opset10::Convert>(gather, decoder->get_output_tensor_type(0));
    res->set_friendly_name(node.get_name());
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
