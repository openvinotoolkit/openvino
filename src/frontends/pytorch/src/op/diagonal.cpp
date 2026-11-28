// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_diagonal(const NodeContext& context) {
    // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
    // Move dim1, dim2 to the end, flatten them, gather flat positions (i + row0) * n2 + (i + col0).
    num_inputs_check(context, 1, 4);
    auto x = context.get_input(0);
    auto get_scalar_input = [&](size_t idx, int64_t default_value) -> Output<Node> {
        if (context.input_is_none(idx)) {
            return context.mark_node(v0::Constant::create(element::i64, Shape{}, {default_value}));
        }
        return context.mark_node(std::make_shared<v0::Convert>(context.get_input(static_cast<int>(idx)), element::i64));
    };
    auto offset = get_scalar_input(1, 0);
    auto dim1 = get_scalar_input(2, 0);
    auto dim2 = get_scalar_input(3, 1);

    Output<Node> shape, rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true, element::i64);
    dim1 = normalize_axis(context, dim1, rank);
    dim2 = normalize_axis(context, dim2, rank);

    auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto minus_one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));

    // Permutation: remaining dims in original order, then dim1, dim2.
    auto axes = context.mark_node(std::make_shared<v4::Range>(zero, rank, one, element::i64));
    auto not_dim1 = context.mark_node(std::make_shared<v1::NotEqual>(axes, dim1));
    auto not_dim2 = context.mark_node(std::make_shared<v1::NotEqual>(axes, dim2));
    auto keep = context.mark_node(std::make_shared<v1::LogicalAnd>(not_dim1, not_dim2));
    Output<Node> batch_axes = context.mark_node(std::make_shared<v3::NonZero>(keep, element::i64));
    batch_axes = context.mark_node(std::make_shared<v1::Reshape>(batch_axes, minus_one_1d, false));
    auto dim1_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(dim1, zero));
    auto dim2_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(dim2, zero));
    auto perm = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_axes, dim1_1d, dim2_1d}, 0));
    auto transposed = context.mark_node(std::make_shared<v1::Transpose>(x, perm));

    // Explicit target shape (no -1), so zero-sized dims stay valid.
    auto batch_shape = context.mark_node(std::make_shared<v8::Gather>(shape, batch_axes, zero));
    auto n1 = context.mark_node(std::make_shared<v8::Gather>(shape, dim1, zero));
    auto n2 = context.mark_node(std::make_shared<v8::Gather>(shape, dim2, zero));
    auto n1_n2 = context.mark_node(std::make_shared<v1::Multiply>(n1, n2));
    auto n1_n2_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(n1_n2, zero));
    auto flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_shape, n1_n2_1d}, 0));
    auto flat = context.mark_node(std::make_shared<v1::Reshape>(transposed, flat_shape, false));

    // offset >= 0: start at (0, offset); offset < 0: start at (-offset, 0).
    auto neg_offset = context.mark_node(std::make_shared<v0::Negative>(offset));
    auto row0 = context.mark_node(std::make_shared<v1::Maximum>(neg_offset, zero));
    auto col0 = context.mark_node(std::make_shared<v1::Maximum>(offset, zero));
    auto rows_left = context.mark_node(std::make_shared<v1::Subtract>(n1, row0));
    auto cols_left = context.mark_node(std::make_shared<v1::Subtract>(n2, col0));
    auto diag_len = context.mark_node(std::make_shared<v1::Minimum>(rows_left, cols_left));
    diag_len = context.mark_node(std::make_shared<v1::Maximum>(diag_len, zero));

    auto steps = context.mark_node(std::make_shared<v4::Range>(zero, diag_len, one, element::i64));
    auto stride = context.mark_node(std::make_shared<v1::Add>(n2, one));
    auto start = context.mark_node(std::make_shared<v1::Multiply>(row0, n2));
    start = context.mark_node(std::make_shared<v1::Add>(start, col0));
    auto indices = context.mark_node(std::make_shared<v1::Multiply>(steps, stride));
    indices = context.mark_node(std::make_shared<v1::Add>(indices, start));

    auto last_axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {-1}));
    return {context.mark_node(std::make_shared<v8::Gather>(flat, indices, last_axis))};
}

}  // namespace ov::frontend::pytorch::op
