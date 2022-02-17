// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_unpack_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto axis = node.get_attribute<int64_t>("axis");
    auto num = node.get_attribute<int64_t>("num");

    auto axis_const = make_shared<Constant>(element::i64, Shape{}, axis);
    auto split = make_shared<Split>(input, axis_const, num);
    set_node_name(node.get_name(), split);
    OutputVector res;
    int i = 0;
    for (auto out : split->outputs()) {
        auto squeezed_res = make_shared<Squeeze>(out, axis_const);
        set_node_name(node.get_name() + "/squeeze_" + to_string(i++), squeezed_res);
        res.push_back(squeezed_res);
    }
    return res;
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
