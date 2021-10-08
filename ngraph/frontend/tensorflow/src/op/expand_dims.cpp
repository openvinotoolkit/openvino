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

OutputVector TranslateExpandDimsOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto dims = node.get_ng_input(1);
    auto unsqueeze = make_shared<Unsqueeze>(input, dims);
    unsqueeze->set_friendly_name(node.get_name());
    return unsqueeze->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph