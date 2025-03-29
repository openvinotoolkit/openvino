// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_delete(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto indices = context.get_input(1);
    if (indices.get_element_type() != element::i32) {
        indices = context.mark_node(std::make_shared<v0::Convert>(indices, element::i32));
    }
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    // calculate the end index for slicing after the target indices
    auto end_index =
        context.mark_node(std::make_shared<v1::Add>(indices, v0::Constant::create(element::i32, Shape{1}, {1})));
    // slice elements before the indices
    auto before_indices =
        context.mark_node(std::make_shared<v1::StridedSlice>(input,
                                                             v0::Constant::create(element::i32, Shape{1}, {0}),
                                                             indices,
                                                             v0::Constant::create(element::i32, Shape{1}, {1}),
                                                             std::vector<int64_t>{1},
                                                             std::vector<int64_t>{0}));
    // slice elements after the indices
    auto after_indices =
        context.mark_node(std::make_shared<v1::StridedSlice>(input,
                                                             end_index,
                                                             input_shape,
                                                             v0::Constant::create(element::i32, Shape{1}, {1}),
                                                             std::vector<int64_t>{0},
                                                             std::vector<int64_t>{1}));
    // add them together
    auto result = context.mark_node(std::make_shared<v0::Concat>(OutputVector{before_indices, after_indices}, 0));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
