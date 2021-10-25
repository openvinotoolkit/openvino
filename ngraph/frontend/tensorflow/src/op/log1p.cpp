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

OutputVector TranslateLog1pOp(const NodeContext& node) {
    auto n = node.get_ng_input(0);
    auto const_1 = make_shared<Constant>(n.get_element_type(), Shape{}, 1);
    auto add = make_shared<Add>(n, const_1);
    auto res = make_shared<Log>(add);
    SetNodeNames(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
