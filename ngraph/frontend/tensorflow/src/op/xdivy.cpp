// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

#if 0

namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateXdivyOp(
        const NodeContext& node) {
    Output<ngraph::Node> ng_x, ng_y;
    TF_RETURN_IF_ERROR(GetInputNodes(ng_op_map, op, ng_x, ng_y));
    auto zero =
            ConstructNgNode<Constant>(node.get_name(), ng_x.get_element_type(),
                                             ngraph::Shape{}, std::vector<int>({0}));
    auto x_is_zero = ConstructNgNode<Equal>(node.get_name(), ng_x, zero);
    auto ng_xdivy = ConstructNgNode<Divide>(node.get_name(), ng_x, ng_y);
    SaveNgOp(ng_op_map, node.get_name(), ConstructNgNode<Select>(
            node.get_name(), x_is_zero, ng_x, ng_xdivy));
    return Status::OK();
}

}
}
#endif