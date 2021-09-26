// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector PlaceholderOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ngraph::element::Type>("dtype");
    auto ng_shape =  node.get_attribute<ngraph::PartialShape>("shape", ngraph::PartialShape());
    return {ConstructNgNode<opset::Parameter>(node.get_name(), ng_et, ng_shape)};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow