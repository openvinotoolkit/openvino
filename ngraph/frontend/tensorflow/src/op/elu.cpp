// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

#include "node_context.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::opset8;

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateEluOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto alpha = node.get_attribute<float>("alpha", 1.0);
    return {ConstructNgNode<Elu>(node.get_name(), input, alpha)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
