// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_linspace_op(const NodeContext& node) {
    // The operation is simple that generates a range [start, ..., stop]
    // with num elements staying in the same distance between each other
    default_op_checks(node, 3, {"LinSpace"});
    auto start = node.get_input(0);
    auto stop = node.get_input(1);
    auto num = node.get_input(2);

    // compute delta value, i.e. distance between neighbor values of the result
    auto const_one = create_same_type_const_scalar<int32_t>(num, 1);
    auto const_zero = create_same_type_const_scalar<int32_t>(num, 0);
    Output<Node> num_minus_one = make_shared<v1::Subtract>(num, const_one);

    // prevent to have num_minus_one to be equal to zero to avoid division by zero
    // that leads to nan values in output
    auto is_num_minus_one_zero = make_shared<v1::Equal>(num_minus_one, const_zero);
    num_minus_one = make_shared<v1::Select>(is_num_minus_one_zero, const_one, num_minus_one);

    // compute delta for evenly-spaced split
    num_minus_one = make_shared<v1::ConvertLike>(num_minus_one, start);
    Output<Node> delta = make_shared<v1::Subtract>(stop, start);
    delta = make_shared<v1::Divide>(delta, num_minus_one);

    // generate a range of numbers [0, 1, ..., num)
    // to have exact numbers of elements equal to num
    Output<Node> range0_n = make_shared<v4::Range>(const_zero, num, const_one, ov::element::f32);
    range0_n = make_shared<v1::ConvertLike>(range0_n, start);

    // compute the result
    Output<Node> linspace = make_shared<v1::Multiply>(range0_n, delta);
    linspace = make_shared<v1::Add>(linspace, start);
    set_node_name(node.get_name(), linspace.get_node_shared_ptr());
    return {linspace};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
