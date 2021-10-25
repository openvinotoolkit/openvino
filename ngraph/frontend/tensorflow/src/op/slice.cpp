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

OutputVector TranslateSliceOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto start = node.get_ng_input(1);
    auto size = node.get_ng_input(2);

    auto stop = make_shared<Add>(start, size);

    auto one = make_shared<Constant>(element::i64, Shape{1}, 1);
    auto shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(one, shape);

    auto res = make_shared<Slice>(input, start, stop, step);
    SetNodeNames(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
