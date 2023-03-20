// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

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
    auto size0 = make_shared<ShapeOf>(s0);
    auto size1 = make_shared<ShapeOf>(s1);
    auto max_size = make_shared<Maximum>(size0, size1);
    auto diff1 = make_shared<Subtract>(max_size, size0);
    auto diff2 = make_shared<Subtract>(max_size, size1);

    // pad the shortest shape value with minus ones
    // to take dynamic shapes into account
    auto const_zero = create_same_type_const<int64_t>(diff1, std::vector<int64_t>{0}, Shape{1});
    auto const_minus_one = create_same_type_const_scalar<int64_t>(s0, -1);
    auto padded_s0 = make_shared<Pad>(s0, diff1, const_zero, const_minus_one, ov::op::PadMode::CONSTANT);
    auto padded_s1 = make_shared<Pad>(s1, diff2, const_zero, const_minus_one, ov::op::PadMode::CONSTANT);

    auto broadcasted_shape = make_shared<Maximum>(padded_s0, padded_s1);
    set_node_name(node.get_name(), broadcasted_shape);
    return {broadcasted_shape};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
