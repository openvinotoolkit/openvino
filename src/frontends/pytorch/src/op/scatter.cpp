// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_scatter(const NodeContext& context) {
    // Out-of-place schema
    // aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor:
    // aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor:
    // aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor:
    // aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor:

    // Inplace schema
    // aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!):
    // aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!):
    // aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!):
    // aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!):
    num_inputs_check(context, 4, 5);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);
    auto index = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::i32));
    auto src = context.get_input(3);

    auto reduction = v12::ScatterElementsUpdate::Reduction::NONE;
    auto input_num = context.get_input_size();
    if (input_num > 4 && !context.input_is_none(input_num - 1)) {
        auto reduce_mode = context.const_input<std::string>(input_num - 1);
        if (reduce_mode == "add") {
            reduction = v12::ScatterElementsUpdate::Reduction::SUM;
        } else if (reduce_mode == "multiply") {
            reduction = v12::ScatterElementsUpdate::Reduction::PROD;
        }
    }
    auto src_partial_shape = src.get_partial_shape();
    auto index_shape_rank = get_shape_rank(context, index);
    auto index_shape = std::get<0>(index_shape_rank);
    auto index_rank = std::get<1>(index_shape_rank);

    // Source input can be either Tensor which should be passed in original shape or Scalar that should be broadcasted
    // into shape of indices.
    // TODO: Figure out way to dynamically broadcast scalar src only, without affecting Tensor src. Current
    // implementation will fail if Scalar source would have dynamic rank.
    if (src_partial_shape.rank().is_static() && src_partial_shape.rank().get_length() == 0) {
        src = context.mark_node(std::make_shared<v3::Broadcast>(src, index_shape));
    }

    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(const_0, index_rank));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(const_1, index_rank));
    // In torch indices can be of different shape than source tensor. Create slice to trim source tensor to shape of
    // indices.
    auto src_pruned = context.mark_node(std::make_shared<v8::Slice>(src, zeros, index_shape, ones));

    auto src_input_dtype = context.mark_node(std::make_shared<v1::ConvertLike>(src_pruned, input));
    return {
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(input, index, src_input_dtype, dim, reduction))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
