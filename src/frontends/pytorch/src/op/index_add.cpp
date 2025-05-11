// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_index_add(const NodeContext& context) {
    // aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
    // aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) ->
    // Tensor(a!)
    num_inputs_check(context, 5, 6);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    auto src = context.get_input(3);
    auto alpha = context.get_input(4);
    auto converted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, src));
    auto alpha_src = context.mark_node(std::make_shared<v1::Multiply>(converted_alpha, src));
    auto input_shape_rank = get_shape_rank(context, input);
    auto const_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_one_0d = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto inp_rank = std::get<1>(input_shape_rank);
    // ScatterElementsUpdate required that index, source and update have the same rank
    // in aten::index_add index represents as 1d-array for specific dim and update may have different size
    // from source in non-indexing axes
    // slice src for having only relevant data
    auto src_broadcast_shape = context.mark_node(std::make_shared<v3::Broadcast>(const_one, inp_rank));
    auto src_broadcasted = context.mark_node(
        std::make_shared<v3::Broadcast>(alpha_src, src_broadcast_shape, BroadcastType::BIDIRECTIONAL));
    auto src_shape_rank = get_shape_rank(context, src_broadcasted);
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto src_rank = std::get<1>(src_shape_rank);
    auto slice_start = context.mark_node(std::make_shared<v3::Broadcast>(const_zero, inp_rank));
    auto axes = get_node_axes_range(context, src_broadcasted);
    auto const_inf =
        context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()}));
    auto slice_end = context.mark_node(std::make_shared<v3::Broadcast>(const_inf, src_rank));
    auto slice_step = context.mark_node(std::make_shared<v3::Broadcast>(const_one, src_rank));
    auto dim_1d = context.mark_node(std::make_shared<v3::Broadcast>(dim, const_one));
    slice_end =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(slice_end,
                                                                       dim_1d,
                                                                       const_one,
                                                                       const_zero,
                                                                       v12::ScatterElementsUpdate::Reduction::NONE));
    auto new_shape = context.mark_node(std::make_shared<v8::Slice>(input, slice_start, slice_end, slice_step, axes));
    new_shape = context.mark_node(std::make_shared<v3::ShapeOf>(new_shape, element::i32));
    auto src_ =
        context.mark_node(std::make_shared<v3::Broadcast>(src_broadcasted, new_shape, BroadcastType::BIDIRECTIONAL));
    auto src_input_dtype = context.mark_node(std::make_shared<v1::ConvertLike>(src_, input));
    // brodcast index to input rank size
    src_rank = context.mark_node(std::make_shared<v3::ShapeOf>(new_shape, element::i32));
    auto new_index_shape = context.mark_node(std::make_shared<v3::Broadcast>(const_one, src_rank));
    auto const_minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    new_index_shape = context.mark_node(
        std::make_shared<v12::ScatterElementsUpdate>(new_index_shape, dim_1d, const_minus_one, const_zero));
    // precerve indicies location for spicifc dim
    auto reshaped_index = context.mark_node(std::make_shared<v1::Reshape>(index, new_index_shape, false));
    auto broadcasted_index =
        context.mark_node(std::make_shared<v3::Broadcast>(reshaped_index, new_shape, BroadcastType::BIDIRECTIONAL));
    auto scatter_result =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(input,
                                                                       broadcasted_index,
                                                                       src_,
                                                                       dim,
                                                                       v12::ScatterElementsUpdate::Reduction::SUM));
    if (!context.input_is_none(5)) {
        context.mutate_input(5, scatter_result);
    }
    return {scatter_result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
