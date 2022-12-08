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

OutputVector translate_pack_op(const NodeContext& node) {
    auto axis = node.get_attribute<int64_t>("axis");
    auto axis_const = make_shared<Constant>(element::i64, Shape{}, axis);

    OutputVector concat_inputs;
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        auto in = node.get_input(static_cast<int>(i));
        concat_inputs.push_back(make_shared<Unsqueeze>(in, axis_const));
    }

    auto res = make_shared<Concat>(concat_inputs, axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
