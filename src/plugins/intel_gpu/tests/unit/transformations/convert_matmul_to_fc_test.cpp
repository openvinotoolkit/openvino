// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>
#include <intel_gpu/op/fully_connected.hpp>
#include <intel_gpu/op/placeholder.hpp>
#include <plugin/transformations/convert_matmul_to_fc.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 1, 2, 2 }, { 1 });
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, true, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 1, 2, 2 }, { 1 });
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(input2, transpose_constant2);

	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto matmul = std::make_shared<op::FullyConnected>(transpose1, transpose2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest2) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1, input2});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest3) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest4) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest5) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{ -1, -1, 2 });
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 3, 2, 2 }, { 1 });
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFullyConnected>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest6) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{ -1, -1, 2 });
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 3, 1, 2 }, { 1 });
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFullyConnected>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest7) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest8) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto fc = std::make_shared<op::FullyConnected>(input1, input2, no_bias);
        auto a_shape = std::make_shared<ov::opset3::ShapeOf>(input1);

        auto I = ov::op::util::node_to_get_shape_value_of_indices_from_shape_node(a_shape, {0, 1});
        auto O = ov::opset1::Constant::create(ov::element::i64, { 1 }, { 3 });
        auto output_shape = std::make_shared<ov::opset1::Concat>(ov::OutputVector{I, O}, 0);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{fc}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest9) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest10) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2, 2 }, { 1 });
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    manager.register_pass<ConvertMatMulToFullyConnected>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest11) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{18, -1, 1});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{18, 80, 1}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFullyConnected>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest12) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1});
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 80, 1}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    manager.register_pass<ConvertMatMulToFullyConnected>();
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest13) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 80, 1}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 80, 1}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest14) {
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
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i8, ov::Shape{1, 80, 1}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest15) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input3 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{64, 32}, {1});

        auto convert = std::make_shared<ov::opset1::Convert>(input3, ov::element::f32);
        ov::mark_as_decompression(convert);

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(input2, convert, false, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul1, matmul2}, ov::ParameterVector{input1, input2});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input3 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{64, 32}, {1});

        auto transpose_constant = ov::opset1::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
        auto transpose = std::make_shared<ov::opset1::Transpose>(input3, transpose_constant);
        auto convert = std::make_shared<ov::opset1::Convert>(transpose, ov::element::f32);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto matmul1 = std::make_shared<op::FullyConnected>(input1, convert, no_bias);
        auto matmul2 = std::make_shared<op::FullyConnected>(input2, convert, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul1, matmul2}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_second_input_rank_adj_1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{5, 2, 3});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 2, 3}, {1});
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_second_input_rank_adj_2) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 2, 3 });
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2, 3 }, { 1 });
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, weights, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 2, 3 });
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 2, 3 }, { 1 });
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, weights, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_second_input_rank_adj_3) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 5, 2, 3 });
        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 1, 2, 3 }, { 1 });
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, weights, false, true);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 5, 2, 3 });

        auto weights = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 1, 2, 3 }, { 1 });
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, weights, no_bias);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_decompress_convert_0) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{ 1, 2, 2 }, { 1 });
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });

        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{ 1, 2, 2 }, { 1 });
        auto transpose_constant = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose = std::make_shared<ov::opset1::Transpose>(input2, transpose_constant);
        auto convert = std::make_shared<ov::opset1::Convert>(transpose, ov::element::f32);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto matmul = std::make_shared<op::FullyConnected>(input1, convert, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_decompress_convert_1) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{ 1, 2, 2 }, { 1 });
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, true, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);

        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{ 1, 2, 2 }, { 1 });
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{ 3 }, { 0, 2, 1 });
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(input2, transpose_constant2);
        auto convert = std::make_shared<ov::opset1::Convert>(transpose2, ov::element::f32);
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto matmul = std::make_shared<op::FullyConnected>(transpose1, convert, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ input1 });
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedTest_compressed_u8_weights) {
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
        manager.register_pass<ConvertMatMulToFullyConnected>();
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
	auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(data, transpose, no_bias);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ matmul }, ov::ParameterVector{ data });
    }
}

// Checked blocked cases
TEST(TransformationTests, ConvertMatMulToFullyConnectedExceptionTest_sibling_matmul_no_convert) {
    auto CreateMatMul = [&](bool mat1_transpose_a, bool mat1_transpose_b,
                            bool mat2_transpose_a, bool mat2_transpose_b) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{64, 32});
        auto input_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{64, 32}, {-1});

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, input_const, mat1_transpose_a, mat1_transpose_b);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(input_const, input_const, mat2_transpose_a, mat2_transpose_b);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul2, matmul1}, ov::ParameterVector{input1});
        return model;
    };

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();

    auto func = CreateMatMul(true, false, true, false);

    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == true);

    func = CreateMatMul(true, false, false, true);
    manager.run_passes(func);
    success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == true);

    func = CreateMatMul(false, true, true, false);
    manager.run_passes(func);
    success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == true);
}

TEST(TransformationTests, ConvertMatMulToFullyConnectedExceptionTest_sibling_matmul) {
    auto CreateMatMul = [&](bool mat1_transpose_a, bool mat1_transpose_b,
                            bool mat2_transpose_a, bool mat2_transpose_b) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{10, 64});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{64, 32});
        auto input3 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{64, 32}, {1});

        auto convert = std::make_shared<ov::opset1::Convert>(input3, ov::element::f32);
        ov::mark_as_decompression(convert);

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, convert, mat1_transpose_a, mat1_transpose_b);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(convert, input2, mat2_transpose_a, mat2_transpose_b);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul1, matmul2}, ov::ParameterVector{input1, input2});
        return model;
    };

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();

    auto func = CreateMatMul(false, false, true, false);

    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);

    func = CreateMatMul(false, false, false, true);
    manager.run_passes(func);
    success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);
}

TEST(TransformationTests, ConvertMatMulToFullyConnectedExceptionTest_sibling_matmul_same_input) {
    auto CreateMatMul = [&](bool mat1_transpose_a, bool mat1_transpose_b,
                            bool mat2_transpose_a, bool mat2_transpose_b) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{64, 32});
        auto input_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64, 32}, {-1});

        auto convert = std::make_shared<ov::opset1::Convert>(input_const, ov::element::f32);
        ov::mark_as_decompression(convert);

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, convert, mat1_transpose_a, mat1_transpose_b);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(convert, convert, mat2_transpose_a, mat2_transpose_b);

        auto model = std::make_shared<ov::Model>(ov::NodeVector{matmul2, matmul1}, ov::ParameterVector{input1});
        return model;
    };

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertMatMulToFullyConnected>();

    auto func = CreateMatMul(true, false, true, false);

    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);

    func = CreateMatMul(true, false, false, true);
    manager.run_passes(func);
    success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);

    func = CreateMatMul(false, true, true, false);
    manager.run_passes(func);
    success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("FullyConnected") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);
}

TEST_F(TransformationTestsF, ConvertMatMulToFullyConnectedExceptionTest) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{64, 32}, {1});

        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(convert, convert, true, false);

        model = std::make_shared<ov::Model>(ov::NodeVector{matmul1, matmul2}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFullyConnected>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 64});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{64, 32}, {1});

        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        auto convert_2 = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);

        auto matmul1 = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);
        auto matmul2 = std::make_shared<ov::opset1::MatMul>(convert_2, convert, true, false);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{matmul1, matmul2}, ov::ParameterVector{input1});
    }
}
