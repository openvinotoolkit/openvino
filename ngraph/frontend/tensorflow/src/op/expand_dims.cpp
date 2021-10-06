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

OutputVector TranslateExpandDimsOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    std::vector<int64_t> dims;
    GetStaticInputVector(node, 1, &dims);
    auto ng_dims = ConstructNgNode<Constant>(node.get_name(), element::i64, ngraph::Shape{dims.size()}, dims);
    return {ConstructNgNode<Unsqueeze>(node.get_name(), ng_input, ng_dims)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph