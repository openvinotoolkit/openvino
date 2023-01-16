// Copyright (C) 2018-2022 Intel Corporation
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
OutputVector translate_list_diff_op(const NodeContext& node) {
    // ListDiff computes the difference between two lists of numbers
    default_op_checks(node, 2, {"ListDiff"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);

    // retrieve attribute
    auto out_idx = node.get_attribute<ov::element::Type>("out_idx", ov::element::i32);

    // unsqueeze both operand to make comparison elements between each other
    auto unsqueeze_x =
        make_shared<Unsqueeze>(x, make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}));
    auto unsqueeze_y =
        make_shared<Unsqueeze>(y, make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}));

    // generate a mask where elements x and y are different
    auto x_is_non_equal_y =
        make_shared<ReduceLogicalOr>(make_shared<NotEqual>(unsqueeze_x, unsqueeze_y),
                                     make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}),
                                     false);
    auto mask01_x_is_non_equal_y =
        make_shared<Select>(x_is_non_equal_y,
                            make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}),
                            make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}));

    // compute indices of x elements different from elements of y
    auto diff_output_indices =
        make_shared<Reshape>(make_shared<NonZero>(mask01_x_is_non_equal_y, out_idx),
                             make_shared<Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}),
                             false);

    // gather elements from x that occur in y
    auto diff_x = make_shared<Gather>(x,
                                      diff_output_indices,
                                      make_shared<Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}));
    return {diff_x, diff_output_indices};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
