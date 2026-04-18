// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace std;
using namespace ov::op;

OutputVector translate_bincount_common(const NodeContext& context,
                                       const Output<Node>& arr,
                                       const Output<Node>& size,
                                       const Output<Node>& weights) {
    auto start = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto step = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto range = context.mark_node(make_shared<v4::Range>(start, size, step, element::i32));

    auto flatten_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto arr_flat = context.mark_node(make_shared<v1::Reshape>(arr, flatten_shape, false));
    auto weights_flat = context.mark_node(make_shared<v1::Reshape>(weights, flatten_shape, false));

    auto axis_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto axis_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto range_col = context.mark_node(make_shared<v0::Unsqueeze>(range, axis_one));
    auto arr_row = context.mark_node(make_shared<v0::Unsqueeze>(arr_flat, axis_zero));
    auto weights_row = context.mark_node(make_shared<v0::Unsqueeze>(weights_flat, axis_zero));

    auto mask = context.mark_node(make_shared<v1::Equal>(range_col, arr_row));
    auto mask_cast = context.mark_node(make_shared<v1::ConvertLike>(mask, weights_row));
    auto weighted = context.mark_node(make_shared<v1::Multiply>(mask_cast, weights_row));
    auto reduce_axis = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto result = context.mark_node(make_shared<v1::ReduceSum>(weighted, reduce_axis));

    return {result};
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
