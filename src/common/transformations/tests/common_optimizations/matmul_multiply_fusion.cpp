// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

TEST_F(TransformationTestsF, MatMulMultiplyFusionConstantWeightsScalarConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{3, 2}, {1, 2, 3, 4, 5, 6});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights);
        auto mul_const = opset8::Constant::create(element::f32, Shape{}, {2});
        auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{3, 2}, {2, 4, 6, 8, 10, 12});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, MatMulMultiplyFusionConstantWeightsNonScalarConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 2, 4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{3, 2}, {1, 2, 3, 4, 5, 6});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1, 1, 1, 2}, {2, 3});
        auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 2, 4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{1, 1, 3, 2}, {2, 6, 6, 12, 10, 18});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, MatMulMultiplyFusionConstantTransposedWeightsNonScalarConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 2, 4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{2, 3}, {1, 2, 3, 4, 5, 6});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, true);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1, 1, 1, 2}, {2, 3});
        auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 2, 4, 3});
        auto weights = opset8::Constant::create(element::f32, Shape{1, 1, 2, 3}, {2, 4, 6, 12, 15, 18});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, true);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, MatMulMultiplyFusionNonConstantTransposedWeightsNonScalarConstant) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3});
        auto weights = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, true);
        auto mul_const = opset8::Constant::create(element::f32, Shape{1, 2}, {4, 5});
        auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data, weights});

        manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3});
        auto weights = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3});
        auto mul_const = opset8::Constant::create(element::f32, Shape{2, 1}, {4, 5});
        auto mul = std::make_shared<opset8::Multiply>(weights, mul_const);
        auto matmul = std::make_shared<opset8::MatMul>(data, mul, false, true);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data, weights});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, MatMulMultiplyFusionNonSingleConsumer) {
    auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3});
    auto weights = opset8::Constant::create(element::f32, Shape{2, 3}, {2, 6, 6, 12, 10, 18});
    auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, true);
    auto mul_const = opset8::Constant::create(element::f32, Shape{1, 2}, {4, 5});
    auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
    auto add = std::make_shared<opset8::Add>(matmul, mul);
    model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data});

    manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

using MatMulMultiplyFusionParams = std::tuple<PartialShape, Shape, bool, Shape, Shape, bool>;

class MatMulMultiplyFusionDynamicShapes : public testing::WithParamInterface<MatMulMultiplyFusionParams>,
                                          public TransformationTestsF {};

TEST_P(MatMulMultiplyFusionDynamicShapes, FusionTest) {
    auto params = GetParam();
    const auto& input_shape = std::get<0>(params);
    const auto& weights_shape = std::get<1>(params);
    bool transpose_b = std::get<2>(params);
    const auto& const_shape = std::get<3>(params);
    const auto& new_weights_shape = std::get<4>(params);
    bool can_fuse = std::get<5>(params);

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto weights = opset8::Constant::create(element::f32, weights_shape, {2});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, transpose_b);
        auto mul_const = opset8::Constant::create(element::f32, const_shape, {4});
        auto mul = std::make_shared<opset8::Multiply>(matmul, mul_const);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulMultiplyFusion>();
    }

    if (can_fuse) {
        auto data = std::make_shared<opset8::Parameter>(element::f32, input_shape);
        auto weights = opset8::Constant::create(element::f32, new_weights_shape, {8});
        auto matmul = std::make_shared<opset8::MatMul>(data, weights, false, transpose_b);
        model_ref = std::make_shared<Model>(NodeVector{matmul}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

std::vector<MatMulMultiplyFusionParams> params = {
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {2, 3}, false, {}, {2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {2, 3}, false, {1}, {2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {2, 3}, false, {1, 3}, {2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {3, 2}, true, {1, 3}, {3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 2, 3}, false, {1, 3}, {4, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 3, 2}, true, {1, 3}, {4, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 2, 3}, false, {1, 1, 3}, {4, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 3, 2}, true, {1, 1, 3}, {4, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 2, 3}, false, {4, 1, 3}, {4, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 3, 2}, true, {4, 1, 3}, {4, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 3, 2, 3}, false, {4, 3, 1, 3}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {4, 3, 3, 2}, true, {4, 3, 1, 3}, {4, 3, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(2), {2, 3}, false, {1, 3}, {2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(2), {3, 2}, true, {1, 3}, {3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {2, 3}, false, {1, 1, 1, 3}, {1, 1, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {3, 2}, true, {1, 1, 1, 3}, {1, 1, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {2, 3}, false, {1, 1, 3}, {1, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {3, 2}, true, {1, 1, 3}, {1, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 2, 3}, false, {1}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 2, 3}, false, {1, 3}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 3, 2}, true, {1, 3}, {4, 3, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 2, 3}, false, {1, 1, 3}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 3, 2}, true, {1, 1, 3}, {4, 3, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 2, 3}, false, {1, 1, 1, 3}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 3, 2}, true, {1, 1, 1, 3}, {4, 3, 3, 2}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 2, 3}, false, {4, 1, 1, 3}, {4, 3, 2, 3}, true),
    MatMulMultiplyFusionParams(PartialShape::dynamic(4), {4, 3, 3, 2}, true, {1, 3, 1, 3}, {4, 3, 3, 2}, true),
    MatMulMultiplyFusionParams({2, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()},
                               {2, 3},
                               false,
                               {2, 1, 1, 3},
                               {2, 1, 2, 3},
                               true),
    MatMulMultiplyFusionParams({Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()},
                               {2, 3},
                               false,
                               {1, 3, 1, 3},
                               {1, 3, 2, 3},
                               true),
    MatMulMultiplyFusionParams({2, 3, Dimension::dynamic(), Dimension::dynamic()},
                               {2, 3},
                               false,
                               {2, 3, 1, 3},
                               {2, 3, 2, 3},
                               true),
    // negative cases
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {2, 3}, false, {1, 1, 1}, {}, false),
    MatMulMultiplyFusionParams(PartialShape::dynamic(2), {2, 3}, false, {1, 1, 1}, {}, false),
    MatMulMultiplyFusionParams(PartialShape::dynamic(), {1, 2, 3}, false, {3, 1, 3}, {}, false),
    MatMulMultiplyFusionParams(PartialShape::dynamic(3), {1, 2, 3}, false, {3, 1, 3}, {}, false),
    MatMulMultiplyFusionParams({1, 1, Dimension::dynamic(), Dimension::dynamic()},
                               {2, 3},
                               false,
                               {2, 3, 1, 3},
                               {},
                               false),
};

INSTANTIATE_TEST_SUITE_P(TransformationTests, MatMulMultiplyFusionDynamicShapes, ::testing::ValuesIn(params));
