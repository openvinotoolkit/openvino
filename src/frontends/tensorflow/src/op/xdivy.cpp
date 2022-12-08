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

OutputVector translate_x_div_y_op(const NodeContext& node) {
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    auto zero = make_shared<Constant>(x.get_element_type(), Shape{}, 0);
    auto x_is_zero = make_shared<Equal>(x, zero);
    auto one = make_shared<Constant>(x.get_element_type(), Shape{}, 1);
    auto select = make_shared<Select>(x_is_zero, one, y);
    auto res = make_shared<Divide>(x, select);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov