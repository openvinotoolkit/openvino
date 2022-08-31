// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_op(const NodeContext& node) {
    auto tf_dtype = node.get_attribute<ov::element::Type>("dtype");
    auto tf_shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape::dynamic());

    auto res = std::make_shared<Parameter>(tf_dtype, tf_shape);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

OutputVector translate_placeholder_with_default_op(const NodeContext& node) {
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