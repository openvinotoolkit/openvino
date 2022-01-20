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
OutputVector translate_select_op(const NodeContext& node) {
    TENSORFLOW_OP_VALIDATION(node, node.get_input_size() == 3, "Select op cannot be converted");
    auto in_1 = node.get_input(0);
    auto in_2 = node.get_input(1);
    auto in_3 = node.get_input(2);
    if (in_1.get_partial_shape().is_static() && in_2.get_partial_shape().is_static()) {
        // select broadcast
        if (in_1.get_shape().size() == 1 && in_2.get_shape().size() > 1) {
            std::vector<uint64_t> axes(in_2.get_shape().size() - 1);
            std::iota(axes.begin(), axes.end(), 1);
            auto unsqueeze_axes = make_shared<Constant>(ov::element::i64, Shape{in_2.get_shape().size() - 1}, axes);
            auto unsqueeze = make_shared<Unsqueeze>(in_1, unsqueeze_axes);
            auto res = make_shared<Select>(unsqueeze, in_2, in_3);
            set_node_name(node.get_name(), res);
            return res->outputs();
        }
    }
    auto res = make_shared<Select>(in_1, in_2, in_3);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
