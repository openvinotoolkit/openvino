// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_list_diff_op(const NodeContext& node) {
    // ListDiff computes the difference between two lists of numbers
    default_op_checks(node, 2, {"ListDiff"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // retrieve attribute
    auto out_idx = node.get_attribute<element::Type>("out_idx", element::i32);

    // unsqueeze both operand to make comparison elements between each other
    auto const_zero = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
    auto unsqueeze_x = make_shared<v0::Unsqueeze>(x, const_one);
    auto unsqueeze_y = make_shared<v0::Unsqueeze>(y, const_zero);

    // generate a mask where elements x and y are different
    // compute 0-1 mask of elements in x that are absent in y
    // 1 means element is absent in y, 0 - otherwise
    auto equal = make_shared<v1::Equal>(unsqueeze_x, unsqueeze_y);
    auto reduce_axis = make_shared<v0::Constant>(element::i32, ov::Shape{}, 1);
    Output<Node> mask_01 = make_shared<v1::ReduceLogicalOr>(equal, reduce_axis, false);
    mask_01 = make_shared<v1::Select>(mask_01, const_zero, const_one);

    // compute indices of x elements different from elements of y
    // compute indices of elements in x that are absent in y
    Output<Node> idx = make_shared<v3::NonZero>(mask_01, out_idx);
    auto new_shape = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    idx = make_shared<v1::Reshape>(idx, new_shape, false);

    // gather elements from x that are absent in y
    auto out = make_shared<v8::Gather>(x, idx, const_zero);

    // set tensor names
    set_out_name({node.get_name() + ":0"}, out);
    set_out_name({node.get_name() + ":1"}, idx);
    return {out, idx};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
