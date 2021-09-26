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

OutputVector TranslateExpandDimsOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    std::vector<int64_t> dims;
    GetStaticInputVector(node, 1, &dims);
    auto ng_dims = ConstructNgNode<opset::Constant>(node.get_name(), element::i64, ngraph::Shape{dims.size()}, dims);
    return {ConstructNgNode<opset::Unsqueeze>(node.get_name(), ng_input, ng_dims)};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow