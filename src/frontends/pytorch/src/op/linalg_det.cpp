// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_linalg_det(const NodeContext& context) {
    // aten::linalg_det(Tensor A) -> Tensor
    // aten::linalg_det.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)
    // aten::det(Tensor self) -> Tensor
    num_inputs_check(context, 1, 2);

    auto input = context.get_input(0);
    const auto et = input.get_element_type();

    // PyTorch documents linalg_det for float/double/cfloat/cdouble only.
    // We support f32/f64; complex is not implemented.
    FRONT_END_OP_CONVERSION_CHECK(et == element::f32 || et == element::f64,
                                  "linalg_det: only float32 and float64 are supported "
                                  "(complex types are not implemented)");

    const auto pshape = input.get_partial_shape();
    const auto rank = pshape.rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static() && rank.get_length() >= 2,
                                  "linalg_det: input rank must be static and >= 2");

    const auto r = rank.get_length();
    const auto n_dim = pshape[r - 1];
    const auto m_dim = pshape[r - 2];

    FRONT_END_OP_CONVERSION_CHECK(n_dim.is_static() && m_dim.is_static(),
                                  "linalg_det: last two dimensions must be static");

    const int64_t n = static_cast<int64_t>(n_dim.get_length());
    const int64_t m = static_cast<int64_t>(m_dim.get_length());

    FRONT_END_OP_CONVERSION_CHECK(n == m, "linalg_det: input must be a square matrix");
    FRONT_END_OP_CONVERSION_CHECK(n >= 1 && n <= 4, "linalg_det: only 1x1 to 4x4 matrices are supported");

    // Gather axes (scalar constants, shared across all extractions)
    auto axis_m2 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {-2}));
    auto axis_m1 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {-1}));

    // Scalar index constants 0..n-1 (shared)
    std::vector<Output<Node>> idx(n);
    for (int64_t k = 0; k < n; ++k) {
        idx[k] = context.mark_node(v0::Constant::create(element::i64, Shape{}, {k}));
    }

    // rows[i] = input[..., i, :] with shape [*batch, n]
    // Gather with scalar index removes the gathered axis
    std::vector<Output<Node>> rows(n);
    for (int64_t i = 0; i < n; ++i) {
        rows[i] = context.mark_node(std::make_shared<v8::Gather>(input, idx[i], axis_m2));
    }

    // a[i][j] = input[..., i, j] with shape [*batch]
    std::vector<std::vector<Output<Node>>> a(n, std::vector<Output<Node>>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            a[i][j] = context.mark_node(std::make_shared<v8::Gather>(rows[i], idx[j], axis_m1));
        }
    }

    // 2x2 determinant of submatrix [[a[r0][c0], a[r0][c1]], [a[r1][c0], a[r1][c1]]]
    auto det2 = [&](int r0, int r1, int c0, int c1) -> Output<Node> {
        auto p1 = context.mark_node(std::make_shared<v1::Multiply>(a[r0][c0], a[r1][c1]));
        auto p2 = context.mark_node(std::make_shared<v1::Multiply>(a[r0][c1], a[r1][c0]));
        return context.mark_node(std::make_shared<v1::Subtract>(p1, p2));
    };

    // 3x3 determinant via cofactor expansion along row r0
    auto det3 = [&](int r0, int r1, int r2, int c0, int c1, int c2) -> Output<Node> {
        auto m00 = det2(r1, r2, c1, c2);
        auto m01 = det2(r1, r2, c0, c2);
        auto m02 = det2(r1, r2, c0, c1);
        auto t0 = context.mark_node(std::make_shared<v1::Multiply>(a[r0][c0], m00));
        auto t1 = context.mark_node(std::make_shared<v1::Multiply>(a[r0][c1], m01));
        auto t2 = context.mark_node(std::make_shared<v1::Multiply>(a[r0][c2], m02));
        return context.mark_node(
            std::make_shared<v1::Add>(context.mark_node(std::make_shared<v1::Subtract>(t0, t1)), t2));
    };

    Output<Node> result;

    if (n == 1) {
        result = a[0][0];
    } else if (n == 2) {
        result = det2(0, 1, 0, 1);
    } else if (n == 3) {
        result = det3(0, 1, 2, 0, 1, 2);
    } else {  // n == 4: cofactor expansion along first row
        auto M00 = det3(1, 2, 3, 1, 2, 3);
        auto M01 = det3(1, 2, 3, 0, 2, 3);
        auto M02 = det3(1, 2, 3, 0, 1, 3);
        auto M03 = det3(1, 2, 3, 0, 1, 2);

        auto t0 = context.mark_node(std::make_shared<v1::Multiply>(a[0][0], M00));
        auto t1 = context.mark_node(std::make_shared<v1::Multiply>(a[0][1], M01));
        auto t2 = context.mark_node(std::make_shared<v1::Multiply>(a[0][2], M02));
        auto t3 = context.mark_node(std::make_shared<v1::Multiply>(a[0][3], M03));

        // det = t0 - t1 + t2 - t3
        auto sub1 = context.mark_node(std::make_shared<v1::Subtract>(t0, t1));
        auto add1 = context.mark_node(std::make_shared<v1::Add>(sub1, t2));
        result = context.mark_node(std::make_shared<v1::Subtract>(add1, t3));
    }

    if (!context.input_is_none(1)) {
        context.mutate_input(1, result);
    }

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
