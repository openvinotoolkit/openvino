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

OutputVector reshape(const ov::frontend::tensorflow_lite::NodeContext& node) {
    size_t input_size = node.get_input_size();
    FRONT_END_GENERAL_CHECK(input_size == 1 || input_size == 2,
                            "Unexpected number of inputs -- ",
                            input_size,
                            ", for node ",
                            node.get_op_type());
    const auto& decoder = get_decoder(node);
    bool has_attribute = decoder->has_attribute(&tflite::ReshapeOptions::new_shape);
    Output<Node> shape;
    if (has_attribute) {
        auto reshape_new_shape = decoder->get_attribute(&tflite::ReshapeOptions::new_shape);
        const auto new_shape = std::vector<int64_t>(reshape_new_shape->begin(), reshape_new_shape->end());
        shape = opset10::Constant::create(element::i64, ov::Shape{new_shape.size()}, new_shape);
    } else {
        FRONT_END_GENERAL_CHECK(
            input_size == 2,
            "Unexpected Reshape operation configuration. No new_shape attribute and no second input");
        shape = node.get_input(1);
    }
    auto reshape = std::make_shared<opset10::Reshape>(node.get_input(0), shape, false);
    reshape->set_friendly_name(node.get_name());
    return reshape->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
