// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include "op_translation_utils.hpp"


using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector reshape(const ov::frontend::tensorflow::NodeContext& node) {
    // convert native attributes to tf appropriate attribute
    const auto& decoder = node.get_decoder();
    size_t input_size = node.get_input_size();
    FRONT_END_GENERAL_CHECK(input_size == 1 || input_size == 2, "Unexpected number of inputs -- ", input_size, ", for node ", decoder->get_op_type());

    Output<Node> shape;
    if (input_size == 1) {
        const auto* reshape_opts = decoder->get_attribute("ReshapeOptions").as<const tflite::ReshapeOptions*>();
        const auto new_shape = std::vector<int64_t>(reshape_opts->new_shape()->begin(), reshape_opts->new_shape()->end());
        shape = opset10::Constant::create(element::i64, ov::Shape{new_shape.size()}, new_shape);
    } else {
        shape = node.get_input(1);
    }
    Output<Node> output = std::make_shared<opset10::Reshape>(node.get_input(0), shape, false);
    return {output};
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov



