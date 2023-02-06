// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_pack_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Pack", "PACK"});
    auto num_size = static_cast<int>(node.get_input_size());

    auto axis = node.get_attribute<int64_t>("axis", 0);
    auto axis_const = make_shared<Constant>(element::i64, Shape{}, axis);

    OutputVector concat_inputs;
    for (int ind = 0; ind < num_size; ++ind) {
        auto in = node.get_input(ind);
        concat_inputs.push_back(make_shared<Unsqueeze>(in, axis_const));
    }

    auto pack = make_shared<Concat>(concat_inputs, axis);
    set_node_name(node.get_name(), pack);
    return {pack};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
