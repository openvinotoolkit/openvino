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

ov::OutputVector translate_size_op(const NodeContext& node) {
    auto data = node.get_input(0);
    auto out_type = node.get_attribute<ov::element::Type>("out_type");
    auto shape_of = make_shared<ShapeOf>(data, out_type);
    auto axis = make_shared<Constant>(ov::element::i64, Shape{}, 0);
    auto res = make_shared<ReduceProd>(shape_of, axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
