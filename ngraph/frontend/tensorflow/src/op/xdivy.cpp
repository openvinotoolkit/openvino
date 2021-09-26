// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateXdivyOp(
        const NodeContext& node) {
    Output<ngraph::Node> ng_x, ng_y;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
    auto zero =
            ConstructNgNode<opset::Constant>(node.get_name(), ng_x.get_element_type(),
                                             ngraph::Shape{}, std::vector<int>({0}));
    auto x_is_zero = ConstructNgNode<opset::Equal>(node.get_name(), ng_x, zero);
    auto ng_xdivy = ConstructNgNode<opset::Divide>(node.get_name(), ng_x, ng_y);
    SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<opset::Select>(
            node.get_name(), x_is_zero, ng_x, ng_xdivy));
    return Status::OK();
}

}
}
#endif