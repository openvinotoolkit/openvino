// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateZerosLikeOp(const NodeContext& node) {
    auto x = node.get_ng_input(0);
    auto shape_of = make_shared<ShapeOf>(x);
    auto zero = make_shared<Constant>(x.get_element_type(), Shape{1}, 0);
    auto broadcast = make_shared<Broadcast>(zero, shape_of);
    broadcast->set_friendly_name(node.get_name());
    return broadcast->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
