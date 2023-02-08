// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_flatten(NodeContext& context) {
    auto start_dim = context.const_input<int64_t>(1);
    auto end_dim = context.const_input<int64_t>(2);

    auto shape = std::make_shared<v3::ShapeOf>(context.get_input(0), element::i32);
    auto rank_ = std::make_shared<v3::ShapeOf>(shape, element::i32);
    auto rank = std::make_shared<v0::Squeeze>(rank_);
    // Use opset::If for dim normalization. For now we only have flatten with constant start and end
    auto start_dim_node = context.get_input(1);
    auto end_dim_node = context.get_input(2);
    if (start_dim < 0) {
        start_dim_node = context.mark_node(std::make_shared<v1::Add>(rank, start_dim_node));
    }
    if (end_dim < 0) {
        end_dim_node = context.mark_node(std::make_shared<v1::Add>(rank, end_dim_node));
    }
    // Slice shape from begin and end, then concat with -1, if slice return empty tensor concat shuold still be able to
    // work with it
    auto zero = v0::Constant::create(element::i32, Shape{1}, {0});
    auto one = v0::Constant::create(element::i32, Shape{1}, {1});
    auto int_max = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
    auto start_dim_u = std::make_shared<v0::Unsqueeze>(start_dim_node, zero);
    auto slice_begin = std::make_shared<v8::Slice>(shape, zero, start_dim_u, one);
    auto neg_1_const = v0::Constant::create(element::i32, Shape{1}, {-1});
    auto end_dim_u = std::make_shared<v0::Unsqueeze>(end_dim_node, zero);
    auto end_dim_next = std::make_shared<v1::Add>(end_dim_u, one);
    auto slice_end = std::make_shared<v8::Slice>(shape, end_dim_next, int_max, one);
    auto new_shape = std::make_shared<v0::Concat>(OutputVector{slice_begin, neg_1_const, slice_end}, 0);

    context.mark_nodes({shape,
                        rank_,
                        rank,
                        zero,
                        one,
                        int_max,
                        start_dim_u,
                        end_dim_u,
                        slice_begin,
                        slice_end,
                        neg_1_const,
                        new_shape});

    return {context.mark_node(std::make_shared<v1::Reshape>(context.get_input(0), new_shape, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov