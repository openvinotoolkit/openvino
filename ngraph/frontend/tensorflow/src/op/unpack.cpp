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

OutputVector TranslateUnpackOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto axis = node.get_attribute<int64_t>("axis");
    auto num = node.get_attribute<int64_t>("num");

    auto axis_const = make_shared<Constant>(element::i64, Shape{}, axis);
    auto split = make_shared<Split>(input, axis_const, num);
    split->set_friendly_name(node.get_name());
    return split->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
