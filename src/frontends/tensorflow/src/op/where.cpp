// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_where_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Where", "WHERE"});
    auto condition = node.get_input(0);
    auto non_zero = make_shared<NonZero>(condition, element::i64);
    auto transpose_order = make_shared<Constant>(element::i32, Shape{2}, vector<int32_t>{1, 0});
    auto res = make_shared<opset8::Transpose>(non_zero, transpose_order);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
