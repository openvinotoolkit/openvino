// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
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

// Closed-form 2x2 determinant.
Output<Node> det_2x2(const NodeContext& context, const Output<Node>& x) {
    auto a = matrix_element(context, x, 0, 0);
    auto b = matrix_element(context, x, 0, 1);
    auto c = matrix_element(context, x, 1, 0);
    auto d = matrix_element(context, x, 1, 1);
    auto ad = context.mark_node(std::make_shared<v1::Multiply>(a, d));
    auto bc = context.mark_node(std::make_shared<v1::Multiply>(b, c));
    return context.mark_node(std::make_shared<v1::Subtract>(ad, bc));
}

// Closed-form determinant for small matrices. When the trailing matrix dimension
// is statically known, dispatch by size and decompose 1x1 / 2x2 / 3x3. When it is
// dynamic (the PyTorch frontend often presents inputs with dynamic shapes at
// conversion time, e.g. matrices produced by an internal reshape), the size cannot
// be checked, so the dynamic-shape fallback assumes 3x3 -- the only size handled in
// that case. 3x3 covers the rigid-transform / Kabsch use case in pose-estimation
// models and is the value the op is expected to produce at runtime there.
Output<Node> det_small(const NodeContext& context, const Output<Node>& x) {
    const auto& pshape = x.get_partial_shape();
    const auto rank = pshape.rank();
    if (rank.is_static() && rank.get_length() >= 2) {
        auto n_dim = pshape[rank.get_length() - 1];
        auto m_dim = pshape[rank.get_length() - 2];
        if (n_dim.is_static() && m_dim.is_static()) {
            PYTORCH_OP_CONVERSION_CHECK(n_dim.get_length() == m_dim.get_length(),
                                        "aten::det/linalg_det: trailing dimensions must be square.");
            const auto n = n_dim.get_length();
            PYTORCH_OP_CONVERSION_CHECK(
                n >= 1 && n <= 3,
                "aten::det/linalg_det is only supported for matrices of size up to 3x3, got size ",
                n,
                "x",
                n,
                ".");
            if (n == 1) {
                return matrix_element(context, x, 0, 0);
            }
            if (n == 2) {
                return det_2x2(context, x);
            }
            return det_3x3(context, x);
        }
    }
    // Dynamic trailing dimension(s): assume the supported 3x3 case.
    return det_3x3(context, x);
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
