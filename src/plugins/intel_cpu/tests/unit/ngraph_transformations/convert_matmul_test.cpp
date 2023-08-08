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
#include <transformations/cpu_opset/common/op/fully_connected.hpp>
#include <transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "transformations/rt_info/decompression.hpp"

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
        auto transpose_constant1 = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 0, 2, 1 });
        auto transpose1 = std::make_shared<ngraph::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 2 }, { 1 });
        auto transpose_constant2 = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{ 2 }, { 1, 0 });
        auto transpose2 = std::make_shared<ngraph::opset1::Transpose>(input2, transpose_constant2);

        auto matmul = std::make_shared<FullyConnectedNode>(transpose1, transpose2, ngraph::Rank(3));

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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 5, 2, 3 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<FullyConnectedNode>(input1, weights,  ngraph::Rank(3));
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_decompress_convert_0) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{ 1, 2, 2 }, { 1 });
        auto convert = std::make_shared<ngraph::opset1::Convert>(input2, ngraph::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, convert, false, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });

        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{ 2, 2 }, { 1 });
        auto transpose_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{ 2 }, { 1, 0 });
        auto transpose = std::make_shared<ngraph::opset1::Transpose>(input2, transpose_constant);
        auto convert = std::make_shared<ngraph::opset1::Convert>(transpose, ngraph::element::f32);

        auto matmul = std::make_shared<FullyConnectedNode>(input1, convert, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_decompress_convert_1) {
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{ 1, 2, 2 }, { 1 });
        auto convert = std::make_shared<ngraph::opset1::Convert>(input2, ngraph::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, convert, true, false);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3, 2, 2 });
        auto transpose_constant1 = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 0, 2, 1 });
        auto transpose1 = std::make_shared<ngraph::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{ 2, 2 }, { 1 });
        auto transpose_constant2 = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{ 2 }, { 1, 0 });
        auto transpose2 = std::make_shared<ngraph::opset1::Transpose>(input2, transpose_constant2);
        auto convert = std::make_shared<ngraph::opset1::Convert>(transpose2, ngraph::element::f32);

        auto matmul = std::make_shared<FullyConnectedNode>(transpose1, convert, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest_compressed_u8_weights) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::u8, ngraph::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ngraph::opset1::Convert>(weights, ngraph::element::f32);
        auto sub_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2}, {1});
        auto sub = std::make_shared<ngraph::opset1::Subtract>(convert, sub_const);
        auto mul_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2}, {1});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(sub, mul_const);
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(data, mul);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{data});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto weights = ngraph::opset1::Constant::create(ngraph::element::u8, ngraph::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ngraph::opset1::Convert>(weights, ngraph::element::f32);
        auto sub_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2}, {1});
        auto sub = std::make_shared<ngraph::opset1::Subtract>(convert, sub_const);
        auto mul_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 2}, {1});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(sub, mul_const);

        auto reshape_const = ngraph::opset1::Constant::create(ov::element::i32, {2}, {2, -1});
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(mul, reshape_const, false);
        auto transpose_const = ngraph::opset1::Constant::create(ov::element::i32, {2}, {1, 0});
        auto transpose = std::make_shared<ngraph::opset1::Transpose>(reshape, transpose_const);
        auto matmul = std::make_shared<FullyConnectedNode>(data, transpose, ngraph::Rank(3));

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ matmul }, ngraph::ParameterVector{ data });
    }
}