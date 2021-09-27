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

OutputVector TranslateCumsumOp(const NodeContext& node) {
    auto ng_x = node.get_ng_input(0), ng_axis = node.get_ng_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive"), reverse = node.get_attribute<bool>("reverse");

    return {ConstructNgNode<opset::CumSum>(node.get_name(), ng_x, ng_axis, exclusive, reverse)};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow