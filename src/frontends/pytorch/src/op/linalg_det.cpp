// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace detail {

// Element x[..., i, j] of a batched matrix, with the two trailing singleton axes
// removed. Indexing the last two axes by position works for both static and
// dynamic shapes (the matrix size need not be known at conversion time).
Output<Node> matrix_element(const NodeContext& context, const Output<Node>& x, int64_t i, int64_t j) {
    auto row_idx = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {i}));
    auto col_idx = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {j}));
    auto axis_row = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    auto axis_col = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto row = context.mark_node(std::make_shared<v8::Gather>(x, row_idx, axis_row));  // (..., 1, n)
    auto elem = context.mark_node(std::make_shared<v8::Gather>(row, col_idx, axis_col));  // (..., 1, 1)
    auto squeeze_axes = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-2, -1}));
    return context.mark_node(std::make_shared<v0::Squeeze>(elem, squeeze_axes));  // (...)
}

// Closed-form determinant of a batched 3x3 matrix via cofactor expansion.
Output<Node> det_3x3(const NodeContext& context, const Output<Node>& x) {
    Output<Node> m[3][3];
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            m[i][j] = matrix_element(context, x, i, j);
        }
    }
    auto mul = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Multiply>(a, b));
    };
    auto sub = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Subtract>(a, b));
    };
    auto c0 = sub(mul(m[1][1], m[2][2]), mul(m[1][2], m[2][1]));
    auto c1 = sub(mul(m[1][0], m[2][2]), mul(m[1][2], m[2][0]));
    auto c2 = sub(mul(m[1][0], m[2][1]), mul(m[1][1], m[2][0]));
    auto t0 = mul(m[0][0], c0);
    auto t1 = mul(m[0][1], c1);
    auto t2 = mul(m[0][2], c2);
    return context.mark_node(std::make_shared<v1::Add>(sub(t0, t1), t2));
}

// Closed-form determinant for 3x3 matrices. The trailing two axes are validated (static
// shapes) or runtime-guarded (dynamic shapes) to be 3x3 by ensure_trailing_square, then the
// cofactor expansion is applied. 3x3 covers the rigid-transform / Kabsch use case in
// pose-estimation models, which is the supported size here.
Output<Node> det_small(const NodeContext& context, const Output<Node>& x) {
    return det_3x3(context, ensure_trailing_square(context, x, 3, "aten::det/linalg_det"));
}

}  // namespace detail

OutputVector translate_linalg_det(const NodeContext& context) {
    // aten::linalg_det(Tensor A) -> Tensor
    // aten::det(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    return {detail::det_small(context, x)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
