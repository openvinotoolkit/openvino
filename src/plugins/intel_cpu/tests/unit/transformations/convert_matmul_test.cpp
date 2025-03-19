// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, ConvertMatMulToFCTest1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, true, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 2}, {1});
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(input2, transpose_constant2);

        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            transpose1,
            transpose2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest2) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1, input2});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest3) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest4) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest5) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2, 2}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest6) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 1, 2}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest7) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest8) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
        auto a_shape = std::make_shared<ov::opset3::ShapeOf>(input1);

        auto I = ov::op::util::node_to_get_shape_value_of_indices_from_shape_node(a_shape, {0, 1});
        auto O = ov::opset1::Constant::create(ov::element::i64, {1}, {3});
        auto output_shape = std::make_shared<ov::opset1::Concat>(ov::OutputVector{I, O}, 0);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest9) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest10) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest11) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{18, -1, 1});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{18, 80, 1}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest12) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 80, 1}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest13) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest14) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i8, ov::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(input1, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(input2, ov::element::f32).get(),
            false,
            true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i8, ov::Shape{1, 80, 1}, {1});

        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_4d_1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3, 4, 5});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{6, 5}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3, 4, 5});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{6, 5}, {1});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_4d_2) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1, 5});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 10, 5}, {1});
        auto fc = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1, 5});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 10, 5}, {1});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_4d_3) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_4d_4) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_4d_5) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3, 2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3, 2, 4});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 5, 4}, {1});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_2) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 3}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, weights, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 3}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            weights,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_3) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, weights, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});

        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            weights,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_decompress_convert_0) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});

        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        auto transpose_constant = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::opset1::Transpose>(convert, transpose_constant);

        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            transpose,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_decompress_convert_1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, true, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(convert, transpose_constant2);

        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            transpose1,
            transpose2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_compressed_u8_weights) {
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto weights = ov::opset1::Constant::create(ov::element::u8, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, ov::element::f32);
        auto sub_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto sub = std::make_shared<ov::opset1::Subtract>(convert, sub_const);
        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);
        auto matmul = std::make_shared<ov::opset1::MatMul>(data, mul);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto weights = ov::opset1::Constant::create(ov::element::u8, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, ov::element::f32);
        auto sub_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto sub = std::make_shared<ov::opset1::Subtract>(convert, sub_const);
        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);

        auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {3}, {0, 2, 1});
        auto transpose = std::make_shared<ov::opset1::Transpose>(mul, transpose_const);
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            data,
            transpose,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{data});
    }
}
