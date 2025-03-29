// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_op(const NodeContext& node) {
    default_op_checks(node, 0, {});

    auto dtype = node.get_attribute<element::Type>("dtype");
    auto shape = node.get_attribute<PartialShape>("shape", PartialShape::dynamic());
    if (shape.rank().is_static() && shape.rank().get_length() == 0 && node.has_attribute("_output_shapes")) {
        // we know some cases when Placeholder operation has empty scalar `shape` attribute value
        // and non-empty `_output_shapes` attribute value.
        // `_output_shapes` attribute value turns to be correct in this case
        auto output_shapes = node.get_attribute<vector<PartialShape>>("_output_shapes");
        if (output_shapes.size() == 1 && output_shapes[0].rank().is_static()) {
            shape = output_shapes[0];
        }
    }

    auto res = make_shared<v0::Parameter>(dtype, shape);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_placeholder_with_default_op(const NodeContext& node) {
    default_op_checks(node, 0, {});

    // For parity with legacy frontend, it creates a constant node with the default value
    // As a rule, PlaceholderWithDefault is mainly used for is_training variables in the model
    TENSORFLOW_OP_VALIDATION(node,
                             node.get_input_size() > 0,
                             "PlaceholderWithDefault must have at least one input that is the default value.");
    auto input = node.get_input(0);
    set_out_name(node.get_name(), input);
    return {input};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
