// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::opset8;
using namespace ngraph::frontend::tf;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {
ngraph::OutputVector TranslateRollOp(const NodeContext& node) {
    auto data = node.get_ng_input(0);
    auto shift = node.get_ng_input(1);
    auto axis = node.get_ng_input(2);
    auto roll = std::make_shared<Roll>(data, shift, axis);
    roll->set_friendly_name(node.get_name());
    return roll->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
