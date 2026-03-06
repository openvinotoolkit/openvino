// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/broadcast.hpp"
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
                                       const Output<Node>& weights,
                                       const element::Type& weights_type) {
    auto start = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int>{0}));
    auto step = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, std::vector<int>{1}));
    auto range = context.mark_node(make_shared<v4::Range>(start, size, step, element::i32));

    auto const_flatten_shape =
        context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, std::vector<int32_t>{-1}));
    auto arr_reshaped = context.mark_node(make_shared<v1::Reshape>(arr, const_flatten_shape, false));
    auto weights_reshaped = context.mark_node(make_shared<v1::Reshape>(weights, const_flatten_shape, false));

    auto const_axis_zero = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, vector<int>({0})));
    auto const_axis_one = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, vector<int>({1})));
    auto unsqueeze_range = context.mark_node(make_shared<v0::Unsqueeze>(range, const_axis_one));
    auto unsqueeze_arr = context.mark_node(make_shared<v0::Unsqueeze>(arr_reshaped, const_axis_zero));
    auto unsqueeze_weights = context.mark_node(make_shared<v0::Unsqueeze>(weights_reshaped, const_axis_zero));

    auto mask = context.mark_node(make_shared<v1::Equal>(unsqueeze_range, unsqueeze_arr));
    auto mask_casted = context.mark_node(make_shared<v1::ConvertLike>(mask, unsqueeze_weights));

    auto to_sum = context.mark_node(make_shared<v1::Multiply>(mask_casted, unsqueeze_weights));
    auto reduce_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 1));
    auto result = context.mark_node(make_shared<v1::ReduceSum>(to_sum, reduce_axis));

    return {result};
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
