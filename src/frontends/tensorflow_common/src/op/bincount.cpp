// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_op_table.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_bincount_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Bincount"});
    auto arr = node.get_input(0);
    auto size = node.get_input(1);
    auto weights = node.get_input(2);

    auto scalar_shape = make_shared<v0::Constant>(element::i32, ov::Shape{0}, std::vector<int32_t>{});
    size = make_shared<v1::Reshape>(size, scalar_shape, false);

    auto weights_type = weights.get_element_type();

    if (weights.get_partial_shape() == ov::Shape{0}) {
        auto arr_shape = make_shared<v3::ShapeOf>(arr, element::i32);
        weights = make_shared<v0::Constant>(weights_type, Shape{}, std::vector<int>{1});
        weights = make_shared<v3::Broadcast>(weights, arr_shape);
    }

    // implementation
    auto start = make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int>{0});
    auto step = make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int>{1});
    auto range = make_shared<v4::Range>(start, size, step, element::i32);

    // Reshape arr and weights to 1D tensors
    auto const_flatten_shape = make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{-1});
    auto arr_reshaped = make_shared<v1::Reshape>(arr, const_flatten_shape, false);
    auto weights_reshaped = make_shared<v1::Reshape>(weights, const_flatten_shape, false);

    // Unsqueeze range to [size, 1] shape and unsqueeze arr and weights to shapes [1, num]
    auto const_axis_zero = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int>({0}));
    auto const_axis_one = make_shared<v0::Constant>(element::i32, Shape{1}, vector<int>({1}));
    auto unsqueeze_range = make_shared<v0::Unsqueeze>(range, const_axis_one);
    auto unsqueeze_arr = make_shared<v0::Unsqueeze>(arr_reshaped, const_axis_zero);
    auto unsqueeze_weights = make_shared<v0::Unsqueeze>(weights_reshaped, const_axis_zero);

    // Generate a mask [size, num] on range == arr
    auto mask = make_shared<v1::Equal>(unsqueeze_range, unsqueeze_arr);
    // Compute the weighted mask
    auto mask_casted = make_shared<v0::Convert>(mask, weights_type);

    auto to_sum = make_shared<v1::Multiply>(mask_casted, unsqueeze_weights);
    auto reduce_axis = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto result = make_shared<v1::ReduceSum>(to_sum, reduce_axis);

    set_node_name(node.get_name(), result);

    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
