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
    auto input = node.get_ng_input(0);
    auto dims = node.get_ng_input(1);
    auto res = make_shared<Unsqueeze>(input, dims);
    SetNodeNames(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov