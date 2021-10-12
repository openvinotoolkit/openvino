// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateExpandDimsOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0);
    std::vector<int64_t> dims;
    GetStaticInputVector(node, 1, &dims);
    auto ng_dims = ConstructNgNode<Constant>(node.get_name(), element::i64, ov::Shape{dims.size()}, dims);
    return {ConstructNgNode<Unsqueeze>(node.get_name(), ng_input, ng_dims)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
