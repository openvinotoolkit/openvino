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
OutputVector translate_broadcast_args_op(const NodeContext& node) {
    default_op_checks(node, 2, {"BroadcastArgs"});
    auto s0 = node.get_input(0);
    auto s1 = node.get_input(1);

    // compute a number of shape elements to append for broadcasting
    auto size0 = make_shared<Squeeze>(make_shared<ShapeOf>(s0));
    auto size1 = make_shared<Squeeze>(make_shared<ShapeOf>(s1));
    auto max_size = make_shared<Maximum>(size0, size1);
    auto diff1 = make_shared<Subtract>(max_size, size0);
    auto diff2 = make_shared<Subtract>(max_size, size1);

    // pad the shortest shape value with minus ones
    // to take dynamic shapes into account
    auto padded_s0 =
        make_shared<Pad>(s0,
                         make_shared<Constant>(diff1->get_element_type(), Shape{1}, std::vector<int64_t>{0}),
                         diff1,
                         make_shared<Constant>(s0.get_element_type(), Shape{}, std::vector<int64_t>{-1}),
                         ov::op::PadMode::CONSTANT);
    auto padded_s1 =
        make_shared<Pad>(s1,
                         make_shared<Constant>(diff2->get_element_type(), Shape{1}, std::vector<int64_t>{0}),
                         diff2,
                         make_shared<Constant>(s1.get_element_type(), Shape{}, std::vector<int64_t>{-1}),
                         ov::op::PadMode::CONSTANT);

    auto broadcasted_shape = make_shared<Maximum>(padded_s0, padded_s1);
    set_node_name(node.get_name(), broadcasted_shape);
    return {broadcasted_shape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov