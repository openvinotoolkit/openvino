// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "plugin/transformations/broadcast_mul_reduce_to_matmul.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// Helper: build the Unsqueeze+Multiply+ReduceSum subgraph
static std::shared_ptr<ov::Model> make_unsq_mul_reduce_model(
        ov::element::Type et,
        const ov::PartialShape& shape_a,
        const ov::PartialShape& shape_b,
        int64_t unsq_axis_a,
        int64_t unsq_axis_b,
        int64_t reduce_axis,
        bool keep_dims = false) {
    auto param_a = std::make_shared<ov::op::v0::Parameter>(et, shape_a);
    auto param_b = std::make_shared<ov::op::v0::Parameter>(et, shape_b);

    auto unsq_a_ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {unsq_axis_a});
    auto unsq_a = std::make_shared<ov::op::v0::Unsqueeze>(param_a, unsq_a_ax);

    auto unsq_b_ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {unsq_axis_b});
    auto unsq_b = std::make_shared<ov::op::v0::Unsqueeze>(param_b, unsq_b_ax);

    auto mul = std::make_shared<ov::op::v1::Multiply>(unsq_a, unsq_b);

    auto red_ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {reduce_axis});
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(mul, red_ax, keep_dims);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{param_a, param_b});
}

// ============================================================================
// Pattern 1 (Mamba2 SSD typical):
//   A:[b,g,L,H,N] --Unsqueeze(3)--> [b,g,L,1,H,N]
//   B:[b,g,L,H,N] --Unsqueeze(2)--> [b,g,1,L,H,N]
//   Multiply --> [b,g,L,L,H,N]
//   ReduceSum(axis=5, keep_dims=false) --> [b,g,L,L,H]
//   == MatMul: [...batch, L, N] x [...batch, N, L]^T -> [...batch, L, L]
// ============================================================================
TEST_F(TransformationTestsF, BroadcastMulReduceToMatMul_Pattern1) {
    // Input model
    {
        model = make_unsq_mul_reduce_model(
            ov::element::f16,
            {1, 1, 256, 64, 128},  // A: [b,g,L,H,N]
            {1, 1, 256, 64, 128},  // B: [b,g,L,H,N]
            3,   // unsqueeze A at axis 3
            2,   // unsqueeze B at axis 2
            5);  // reduce axis 5
        manager.register_pass<BroadcastMulReduceToMatMul>();
    }
    // Reference model: Transpose_A + Transpose_B + MatMul(transpose_b=true) + Transpose_out
    {
        auto param_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, 256, 64, 128});
        auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, 256, 64, 128});

        // A: [b,g,L,H,N] -> [...batch, M, K] = [b,g,H, L, N]  (batch={0,1,4}, M=axis_b=2->L, K=axis_r=5->N)
        // Expanded axes: batch={0,1,4}, axis_b=3(M), axis_a=2(N), axis_r=5(K)
        // map_to_a: expanded -> skip axis_a=3 -> {0,1,2,3,4} mapped from {0,1,4,3,5} but let me compute properly.
        // axis_a=3 (unsq axis for A), axis_b=2 (unsq axis for B), axis_r=5 (reduce)
        // batch in expanded: {0,1,4}
        // map_to_a(d) = d if d<3, d-1 if d>=3 (skip axis_a=3)
        // transpose_a: [map_to_a(0), map_to_a(1), map_to_a(4), map_to_a(2), map_to_a(5)]
        //            = [0, 1, 3, 2, 4]
        auto perm_a = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 1, 3, 2, 4});
        auto tr_a = std::make_shared<ov::op::v1::Transpose>(param_a, perm_a);

        // B: [b,g,L,H,N] -> [...batch, N, K] = [b,g,H, L, N]
        // map_to_b(d) = d if d<2, d-1 if d>=2 (skip axis_b=2)
        // transpose_b: [map_to_b(0), map_to_b(1), map_to_b(4), map_to_b(3), map_to_b(5)]
        //   but wait, axis_a=3 in expanded maps to B's N dim
        //   transpose_b order: [batch..., map_to_b(axis_a), map_to_b(axis_r)]
        //   = [map_to_b(0), map_to_b(1), map_to_b(4), map_to_b(3), map_to_b(5)]
        //   = [0, 1, 3, 2, 4]
        auto perm_b = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 1, 3, 2, 4});
        auto tr_b = std::make_shared<ov::op::v1::Transpose>(param_b, perm_b);

        // MatMul(transpose_b=true): [b,g,H, L, N] x [b,g,H, L, N]^T -> [b,g,H, L, L]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(tr_a, tr_b, false, true);

        // Output transpose: MatMul output is [...batch, M, N] in expanded labeling = [b(0),g(1),H(4), L(3), L(2)]
        // Need expected order: remove axis_r(5) -> [0,1,2,3,4] -> which maps to expanded [0,1,2,3,4]
        // matmul_output_order = [batch..., axis_b, axis_a] = [0, 1, 4, 3, 2]
        //   (positions in expanded: batch axes are {0,1,4} in the original expanded rank-6 space,
        //    then axis_b=3(M), axis_a=2(N) -- but in the rank-5 output we need to figure the mapping)
        // Actually, the pass computes: matmul_output_order = [0,1,4, 3, 2]
        //                              expected_output_order = [0,1,2,3,4]
        // transpose_out[i] = j where matmul_output_order[j] == expected_output_order[i]
        // expected[0]=0 -> matmul[0]=0 -> j=0; expected[1]=1 -> matmul[1]=1 -> j=1;
        // expected[2]=2 -> matmul[4]=2 -> j=4; expected[3]=3 -> matmul[3]=3 -> j=3;
        // expected[4]=4 -> matmul[2]=4 -> j=2;
        // transpose_out = [0, 1, 4, 3, 2]
        auto perm_out = ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{0, 1, 4, 3, 2});
        auto tr_out = std::make_shared<ov::op::v1::Transpose>(matmul, perm_out);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{tr_out}, ov::ParameterVector{param_a, param_b});
    }
}

// ============================================================================
// Simple 3D case: no transpose needed for either input or output
//   A:[M,K] --Unsqueeze(1)--> [M,1,K]
//   B:[N,K] --Unsqueeze(0)--> [1,N,K]
//   Multiply --> [M,N,K]
//   ReduceSum(axis=2) --> [M,N]
//   == MatMul(A, B^T): [M,K] x [N,K]^T -> [M,N]
// ============================================================================
TEST_F(TransformationTestsF, BroadcastMulReduceToMatMul_Simple3D) {
    {
        model = make_unsq_mul_reduce_model(
            ov::element::f32,
            {8, 16},   // A: [M, K]
            {4, 16},   // B: [N, K]
            1,   // unsqueeze A at axis 1
            0,   // unsqueeze B at axis 0
            2);  // reduce axis 2 (K)
        manager.register_pass<BroadcastMulReduceToMatMul>();
    }
    {
        // Expanded: axis_a=1, axis_b=0, axis_r=2
        // batch_axes: none (all 3 dims are axis_a, axis_b, axis_r)
        // A original: [M, K] (rank 2), map_to_a skips axis_a=1: d<1->d, d>=1->d-1
        // transpose_a = [map_to_a(axis_b=0), map_to_a(axis_r=2)] = [0, 1] -> identity, no transpose
        // B original: [N, K] (rank 2), map_to_b skips axis_b=0: d<0->d, d>=0->d-1
        // transpose_b = [map_to_b(axis_a=1), map_to_b(axis_r=2)] = [0, 1] -> identity, no transpose
        // MatMul output order: [axis_b=0, axis_a=1], expected: [0, 1] -> identity, no output transpose
        auto param_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 16});
        auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 16});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, false, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{param_a, param_b});
    }
}

// ============================================================================
// Batched 4D case with output transpose
//   A:[B,M,K] --Unsqueeze(2)--> [B,M,1,K]
//   B:[B,N,K] --Unsqueeze(1)--> [B,1,N,K]
//   Multiply --> [B,M,N,K]
//   ReduceSum(axis=3) --> [B,M,N]
//   == MatMul: [B,M,K] x [B,N,K]^T -> [B,M,N]
// ============================================================================
TEST_F(TransformationTestsF, BroadcastMulReduceToMatMul_Batched4D) {
    {
        model = make_unsq_mul_reduce_model(
            ov::element::f32,
            {2, 8, 16},  // A: [B, M, K]
            {2, 4, 16},  // B: [B, N, K]
            2,   // unsqueeze A at axis 2
            1,   // unsqueeze B at axis 1
            3);  // reduce axis 3 (K)
        manager.register_pass<BroadcastMulReduceToMatMul>();
    }
    {
        // Expanded: axis_a=2, axis_b=1, axis_r=3, batch={0}
        // transpose_a = [map_to_a(0), map_to_a(1), map_to_a(3)] = [0, 1, 2] -> identity
        // transpose_b = [map_to_b(0), map_to_b(2), map_to_b(3)] = [0, 1, 2] -> identity
        // matmul_output_order = [0, 1, 2], expected = [0, 1, 2] -> identity
        auto param_a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 8, 16});
        auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 4, 16});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param_a, param_b, false, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{param_a, param_b});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
