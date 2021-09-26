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

OutputVector TranslateCastOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);

    auto ng_et = node.get_attribute<element::Type>("DstT");
    return {ConstructNgNode<opset::Convert>(node.get_name(), ng_input, ng_et)};
}

}  // namespace ngraph_bridge
}  // namespace tensorflow