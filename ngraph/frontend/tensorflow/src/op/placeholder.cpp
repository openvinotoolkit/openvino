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

OutputVector PlaceholderOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ngraph::element::Type>("dtype");
    auto ng_shape = node.get_attribute<ngraph::PartialShape>("shape", ngraph::PartialShape());
    return {ConstructNgNode<Parameter>(node.get_name(), ng_et, ng_shape)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
