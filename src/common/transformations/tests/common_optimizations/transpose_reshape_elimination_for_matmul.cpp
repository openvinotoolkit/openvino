// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, TransposeReshapeEliminationForMatMul) {
    Shape data_shape_1{10, 2};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 2, 0});
        auto transpose_before = std::make_shared<ov::op::v1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = ov::op::v0::Constant::create(element::i32, Shape{2}, {2, 250});
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, reshape_before);
        auto const_reshape_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {2, 0, 1});
        auto tranpose_after = std::make_shared<ov::op::v1::Transpose>(reshape_after, const_tranpose_after);
        model = std::make_shared<Model>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::TransposeReshapeEliminationForMatmul>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, data_2);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, TransposeReshapeEliminationForMatMul_TransposedA) {
    Shape data_shape_1{2, 10};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 2, 0});
        auto transpose_before = std::make_shared<ov::op::v1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = ov::op::v0::Constant::create(element::i32, Shape{2}, {2, 250});
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, reshape_before, true, false);
        auto const_reshape_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {2, 0, 1});
        auto tranpose_after = std::make_shared<ov::op::v1::Transpose>(reshape_after, const_tranpose_after);
        model = std::make_shared<Model>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::TransposeReshapeEliminationForMatmul>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, data_2, true, false);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, TransposeReshapeEliminationForMatMul_TransposedB) {
    Shape data_shape_1{10, 2};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = ov::op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1});
        auto transpose_before = std::make_shared<ov::op::v1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = ov::op::v0::Constant::create(element::i32, Shape{2}, {250, 2});
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, reshape_before, false, true);
        auto const_reshape_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
        auto tranpose_after = std::make_shared<ov::op::v1::Transpose>(reshape_after, const_tranpose_after);
        model = std::make_shared<Model>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::TransposeReshapeEliminationForMatmul>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, data_2);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, TransposeReshapeEliminationForMatMul_TransposedAB) {
    Shape data_shape_1{2, 10};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = ov::op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1});
        auto transpose_before = std::make_shared<ov::op::v1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = ov::op::v0::Constant::create(element::i32, Shape{2}, {250, 2});
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, reshape_before, true, true);
        auto const_reshape_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
        auto tranpose_after = std::make_shared<ov::op::v1::Transpose>(reshape_after, const_tranpose_after);
        model = std::make_shared<Model>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::TransposeReshapeEliminationForMatmul>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data_1, data_2, true, false);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, TransposeReshapeEliminationForMatMul_Einsum) {
    Shape data_shape_1{5, 2};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2}, "kl,mlj->mkj");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::EinsumDecomposition>();
        manager.register_pass<ov::pass::TransposeReshapeEliminationForMatmul>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto broadcast_shape_constant_1 =
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{data_shape_1.size()}, data_shape_1);
        auto broadcast_shape_constant_2 =
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{data_shape_2.size()}, data_shape_2);
        auto broadcast_1 = std::make_shared<ov::op::v3::Broadcast>(data_1,
                                                                   broadcast_shape_constant_1,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        auto broadcast_2 = std::make_shared<ov::op::v3::Broadcast>(data_2,
                                                                   broadcast_shape_constant_2,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        // for some cases Reshape may be first input for Matmul
        auto shape_constant =
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{data_shape_1.size()}, data_shape_1);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast_1, shape_constant, false);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape, broadcast_2, false, false);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }
}
