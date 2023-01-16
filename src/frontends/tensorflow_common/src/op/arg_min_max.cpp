// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_arg_min_max(const NodeContext& node, std::string mode) {
    auto input = node.get_input(0);

    // TensorFlow uses axis with default value equal to zero
    int64_t axis = 0;
    if (node.get_input_size() > 1) {
        TENSORFLOW_OP_VALIDATION(node,
                                 std::dynamic_pointer_cast<opset8::Constant>(node.get_input(1).get_node_shared_ptr()),
                                 "ArgMax/ArgMin is not supported with non-constant axis input");
        std::vector<int64_t> axes;
        get_const_input(node, 1, &axes);
        TENSORFLOW_OP_VALIDATION(node, axes.size() == 1, "ArgMax/ArgMin must be with a scalar axis input.");
        axis = axes[0];
    }
    auto output_type = node.get_attribute<element::Type>("output_type", element::i64);

    // compute indices of max/min values using TopK
    auto k = make_shared<Constant>(element::i64, Shape{}, 1);
    // TODO: define sort attribute for TensorFlow case
    auto top_k = std::make_shared<TopK>(input, k, axis, mode, "none", output_type);

    auto axis_to_remove = make_shared<Constant>(element::i64, Shape{1}, std::vector<int64_t>({axis}));
    auto res = make_shared<Squeeze>(top_k->output(1), axis_to_remove);
    set_node_name(node.get_name(), res);
    return {res};
}

OutputVector translate_arg_max_op(const NodeContext& node) {
    return (translate_arg_min_max(node, "max"));
}

OutputVector translate_arg_min_op(const NodeContext& node) {
    return (translate_arg_min_max(node, "min"));
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
