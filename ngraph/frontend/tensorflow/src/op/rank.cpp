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

ngraph::OutputVector TranslateRankOp(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto shape_of_1 = make_shared<ShapeOf>(data, ngraph::element::i64);
    auto shape_of_2 = make_shared<ShapeOf>(shape_of_1, ngraph::element::i64);
    shape_of_2->set_friendly_name(node.get_name());
    return shape_of_2->outputs();
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
