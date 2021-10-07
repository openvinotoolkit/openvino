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

ngraph::OutputVector TranslateSizeOp(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto out_type = node.get_attribute<ngraph::element::Type>("out_type");
    auto shape_of = make_shared<ShapeOf>(data, out_type);
    auto axis = make_shared<Constant>(ngraph::element::i64, Shape{}, 0);
    auto size = make_shared<ReduceProd>(shape_of, axis);
    size->set_friendly_name(node.get_name());
    return size->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
