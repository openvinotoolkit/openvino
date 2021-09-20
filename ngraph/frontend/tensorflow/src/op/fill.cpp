// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <default_opset.h>
#include <tensorflow_frontend/node_context.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateFillOp(const NodeContext& node) {
    auto ng_dims = node.get_ng_input(0), ng_value = node.get_ng_input(1);
    return {ConstructNgNode<opset::Broadcast>(node.get_name(), ng_value, ng_dims)};
}
}
}
