// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>
#include <openvino/core/model.hpp>
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include <ov_ops/type_relaxed.hpp>
#include <transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/fully_connected.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/random_uniform.hpp"

using namespace testing;
using namespace ov::intel_cpu;

class ConvertMatMulToFCTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ConvertMatMulToFC>();
    }
};

struct MatMulToFCParam {
    std::string name;
    ov::PartialShape input_shape;
    ov::Shape weights_shape;
    bool transpose_a;
    bool transpose_b;
    ov::element::Type output_type;
};

class ConvertMatMulToFCParamTests : public ConvertMatMulToFCTests,
                                    public WithParamInterface<MatMulToFCParam> {
public:
    static std::string getTestCaseName(const TestParamInfo<MatMulToFCParam>& info) {
        return info.param.name;
    }
};

TEST_P(ConvertMatMulToFCParamTests, CompareWithRef) {
    const auto& p = GetParam();
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p.input_shape);
        auto input2 = ov::opset1::Constant::create(ov::element::f32, p.weights_shape, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, p.transpose_a, p.transpose_b);
        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
    {
        auto make_transpose = [](const std::shared_ptr<ov::Node>& node, size_t rank) {
            std::vector<int32_t> order(rank);
            for (size_t i = 0; i < rank; ++i)
                order[i] = static_cast<int32_t>(i);
            std::swap(order[rank - 1], order[rank - 2]);
            auto order_const = ov::opset1::Constant::create(ov::element::i32, ov::Shape{rank}, order);
            return std::make_shared<ov::opset1::Transpose>(node, order_const);
        };

        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p.input_shape);
        auto input2 = ov::opset1::Constant::create(ov::element::f32, p.weights_shape, {1});

        std::shared_ptr<ov::Node> fc_input = input1;
        std::shared_ptr<ov::Node> fc_weights = input2;

        if (p.transpose_a) {
            fc_input = make_transpose(input1, p.input_shape.rank().get_length());
        }
        if (!p.transpose_b) {
            fc_weights = make_transpose(input2, p.weights_shape.size());
        }

        std::shared_ptr<ov::Node> fc;
        if (p.output_type != ov::element::dynamic) {
            fc = std::make_shared<ov::op::internal::FullyConnected>(
                fc_input, fc_weights,
                std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
                p.output_type);
        } else {
            fc = std::make_shared<ov::op::internal::FullyConnected>(
                fc_input, fc_weights,
                std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));
        }
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
    }
}

INSTANTIATE_TEST_SUITE_P(ConvertMatMulToFC, ConvertMatMulToFCParamTests,
    ::testing::Values(
        // transpose_b=true: MatMul(A, B, false, true) -> FC(A, B, bias)
        MatMulToFCParam{"3d_static_2x2", {3, 2, 2}, {2, 2}, false, true, ov::element::dynamic},
        MatMulToFCParam{"3d_dynamic_2x2", {-1, -1, 2}, {2, 2}, false, true, ov::element::dynamic},
        MatMulToFCParam{"3d_static_3x2", {3, 2, 2}, {3, 2}, false, true, ov::element::dynamic},
        MatMulToFCParam{"3d_dynamic_3x2", {-1, -1, 2}, {3, 2}, false, true, ov::element::dynamic},
        MatMulToFCParam{"3d_dynamic_80x1", {-1, -1, 1}, {1, 80, 1}, false, true, ov::element::dynamic},
        MatMulToFCParam{"4d_dynamic_10x5", {-1, -1, 1, 5}, {1, 10, 5}, false, true, ov::element::dynamic},
        MatMulToFCParam{"3d_batch_dim", {5, 2, 3}, {1, 2, 3}, false, true, ov::element::dynamic},
        MatMulToFCParam{"2d_equal", {2, 3}, {2, 3}, false, true, ov::element::dynamic},
        MatMulToFCParam{"4d_static_6x5", {2, 3, 4, 5}, {6, 5}, false, true, ov::element::f32},
        MatMulToFCParam{"4d_2d_to_4d", {2, 4}, {1, 1, 5, 4}, false, true, ov::element::f32},
        MatMulToFCParam{"4d_3d_to_4d", {3, 2, 4}, {1, 1, 5, 4}, false, true, ov::element::f32},
        MatMulToFCParam{"4d_4d_to_4d", {2, 3, 2, 4}, {1, 1, 5, 4}, false, true, ov::element::f32},
        // transpose_a=true: MatMul(A, B, true, false) -> FC(Transpose(A), Transpose(B), bias)
        MatMulToFCParam{"3d_transpose_a", {3, 2, 2}, {1, 2, 2}, true, false, ov::element::dynamic}),
    ConvertMatMulToFCParamTests::getTestCaseName);

TEST_F(ConvertMatMulToFCTests, BothParamsNotConverted) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1, input2});
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(ConvertMatMulToFCTests, FullyDynamicRankNotConverted) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

    model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
}

TEST_F(ConvertMatMulToFCTests, TypeRelaxedU8I8) {
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

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i8, ov::Shape{1, 80, 1}, {1});

        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}),
            ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(ConvertMatMulToFCTests, DecompressF16Weights) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, false, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
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

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(ConvertMatMulToFCTests, DecompressF16Weights_TransposeA) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(input2, ov::element::f32);
        ov::mark_as_decompression(convert);
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, convert, true, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
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

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(ConvertMatMulToFCTests, CompressedU8WeightsWithSubMul) {
    {
        auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{3, 2, 2});
        auto weights = ov::opset1::Constant::create(ov::element::u8, ov::Shape{1, 2, 2}, {1});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, ov::element::f32);
        auto sub_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto sub = std::make_shared<ov::opset1::Subtract>(convert, sub_const);
        auto mul_const = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(sub, mul_const);
        auto matmul = std::make_shared<ov::opset1::MatMul>(data, mul);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data});
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

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data});
    }
}

TEST_F(ConvertMatMulToFCTests, RandomUniformWeightsNotConverted) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});

        auto random_uniform_shape = ov::opset1::Constant::create(ov::element::i32, ov::Shape{2}, {2, 2});
        auto random_uniform_min = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {0.0});
        auto random_uniform_max = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {1.0});
        auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(random_uniform_shape,
                                                                          random_uniform_min,
                                                                          random_uniform_max,
                                                                          ov::element::f32);

        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, random_uniform, false, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});

        auto random_uniform_shape = ov::opset1::Constant::create(ov::element::i32, ov::Shape{2}, {2, 2});
        auto random_uniform_min = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {0.0});
        auto random_uniform_max = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {1.0});
        auto random_uniform = std::make_shared<ov::op::v8::RandomUniform>(random_uniform_shape,
                                                                          random_uniform_min,
                                                                          random_uniform_max,
                                                                          ov::element::f32);

        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, random_uniform, false, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}
