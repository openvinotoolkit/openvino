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

OutputVector translate_where_op(const NodeContext& node) {
    auto x = node.get_input(0);
    auto non_zero = make_shared<NonZero>(x);
    auto transpose_order = make_shared<Constant>(element::i64, Shape{2}, vector<int64_t>{1, 0});
    auto res = make_shared<opset8::Transpose>(non_zero, transpose_order);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
