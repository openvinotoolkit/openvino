// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_linked_op(const NodeContext& node) {
    auto dtype = node.get_attribute<ov::element::Type>("dtype");
    auto shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape::dynamic());
    auto translate_session = node.get_translate_session();
    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    auto model = reinterpret_cast<ov::frontend::tensorflow::InputModel*>(translate_session->get_input_model().get());
    auto tensor_places = model->get_tensor_places();
    auto saved_model_input_names = model->get_saved_model_input_names();

    if (saved_model_input_names.get() && saved_model_input_names->size() > 0) {
        auto input_name = saved_model_input_names->find(node.get_name());
        if (input_name == saved_model_input_names->end()) {
            input_name = saved_model_input_names->find(node.get_name() + ":0");
        }
        if (input_name != saved_model_input_names->end()) {
            auto tensor_place = tensor_places.find(input_name->second);
            if (tensor_place != tensor_places.end()) {
                shape = tensor_place->second->get_partial_shape();
            }
        }
    }

    if (shape.rank().is_static() && shape.rank().get_length() == 0 && node.has_attribute("_output_shapes")) {
        // we know some cases when Placeholder operation has empty scalar `shape` attribute value
        // and non-empty `_output_shapes` attribute value.
        // `_output_shapes` attribute value turns to be correct in this case
        auto output_shapes = node.get_attribute<std::vector<ov::PartialShape>>("_output_shapes");
        if (output_shapes.size() == 1 && output_shapes[0].rank().is_static()) {
            shape = output_shapes[0];
        }
    }

    auto res = std::make_shared<Parameter>(dtype, shape);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
