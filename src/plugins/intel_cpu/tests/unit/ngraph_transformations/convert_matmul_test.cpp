// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph_transformations/op/fully_connected.hpp>
#include <ngraph_transformations/convert_matmul_to_fc.hpp>
#include <ngraph_transformations/fc_bias_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, ConvertMatMulToFCTest1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 2, 2 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, true, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto transpose_constant = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 0, 2, 1 });
        auto transpose = std::make_shared<ngraph::opset1::Transpose>(input1, transpose_constant);
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 2 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(transpose, input2, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest2) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest3) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest4) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest5) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{ -1, -1, 2 });
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest6) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{ -1, -1, 2 });
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest7) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest8) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 2}, {1});

        auto fc = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));
        auto a_shape = std::make_shared<ngraph::opset3::ShapeOf>(input1);

        auto I = ov::op::util::node_to_get_shape_value_of_indices_from_shape_node(a_shape, {0, 1});
        auto O = ngraph::opset1::Constant::create(ngraph::element::i64, { 1 }, { 3 });
        auto output_shape = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{I, O}, 0);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest9) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest10) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 2 }, { 1 });
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest11) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{18, -1, 1});
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{18, 80, 1}, {1});
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest12) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, -1, 1});
    auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 80, 1}, {1});
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFC>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest13) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 1});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 1});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{80, 1}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest14) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::PartialShape{-1, -1, 1});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::MatMul>>(
            ov::element::TypeVector{ngraph::element::f32, ngraph::element::f32},
            ov::element::TypeVector{ngraph::element::f32},
            ov::op::TemporaryReplaceOutputType(input1, ngraph::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(input2, ngraph::element::f32).get(),
            false,
            true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::PartialShape{-1, -1, 1});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{80, 1}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(3), ngraph::element::f32);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(3));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest2) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(3));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, -1, 3072});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 3072}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest3) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(2));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest4) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 786}, {1});
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input1});
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{-1, 128});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786, 128}, {1});
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(2));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fc}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest5) {
    auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 786, 128 }, { 1 });
    auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

    auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 786 }, { 1 });
    auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });
    manager.register_pass<FullyConnectedBiasFusion>();
}

TEST_F(TransformationTestsF, FullyConnectedBiasFusionTest6) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::PartialShape{ -1, -1 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{ 786, 128 }, { 1 });
        auto fc = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2), ngraph::element::f32);

        auto const_bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 786 }, { 1 });
        auto add = std::make_shared<ngraph::opset1::Add>(fc, const_bias);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::PartialShape{ -1, -1 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{ 786, 128 }, { 1 });
        auto bias = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{786}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights, bias, ngraph::Rank(2), ngraph::element::f32);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{5, 2, 3});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 3}, {1});
        auto matmul = std::make_shared<FullyConnectedNode>(input1, input2, ngraph::Rank(2));
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_2) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, weights, false, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights, ngraph::Rank(2));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_second_input_rank_adj_3) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 5, 2, 3 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 2, 3 }, { 1 });
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, weights, false, true);
        auto biases = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 1, 2 }, { 1 });
        auto add = std::make_shared<ngraph::opset1::Add>(matmul, biases);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
        manager.register_pass<FullyConnectedBiasFusion>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 5, 2, 3 });
        auto reshape_before_const = ngraph::opset1::Constant::create(ngraph::element::i64, { 2 }, { -1, 3 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto biases = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights, biases, ngraph::Rank(2));
        auto reshape_after_const = ngraph::opset1::Constant::create(ngraph::element::i64, { 4 }, { 1, 5, 2, 2 });

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}
