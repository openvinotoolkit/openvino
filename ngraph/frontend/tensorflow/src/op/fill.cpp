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

OutputVector TranslateFillOp(const NodeContext& node) {
    auto ng_dims = node.get_ng_input(0);
    auto ng_value = node.get_ng_input(1);
    return {ConstructNgNode<Broadcast>(node.get_name(), ng_value, ng_dims)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
