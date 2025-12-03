// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_gather(const NodeContext& context) {
    // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
    // aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 5, true);  // allow_complex = true
    auto [x, complex] = unwrap_complex(context.get_input(0));

    auto axis = context.const_input<int64_t>(1);
    auto index = context.get_input(2);
    index = context.mark_node(std::make_shared<v0::Convert>(index, element::i32));

    // For complex tensors, GatherElements requires data and indices to have the same rank.
    // The underlying data has an extra dimension (2) at the end, so we need to expand indices.
    if (complex) {
        // Add extra dimension at the end: [N, M] -> [N, M, 1]
        auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
        index = context.mark_node(std::make_shared<v0::Unsqueeze>(index, minus_one));
        // Broadcast to match data shape's last dimension (2)
        // Get shape of index and append 2 to it
        auto index_shape = context.mark_node(std::make_shared<v3::ShapeOf>(index, element::i32));
        auto two = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
        auto slice_begin = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        auto slice_end = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        auto step = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
        auto shape_without_last = context.mark_node(std::make_shared<v8::Slice>(index_shape, slice_begin, slice_end, step));
        auto target_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{shape_without_last, two}, 0));
        index = context.mark_node(std::make_shared<v3::Broadcast>(index, target_shape));
    }

    // input 3 sparse_grad if True, gradient w.r.t. input will be a sparse tensor, used only for training, skip
    auto gather_elements = context.mark_node(std::make_shared<v6::GatherElements>(x, index, axis));
    if (!context.input_is_none(4)) {
        context.mutate_input(4, gather_elements);
    }

    return {wrap_complex(context, gather_elements, complex)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
