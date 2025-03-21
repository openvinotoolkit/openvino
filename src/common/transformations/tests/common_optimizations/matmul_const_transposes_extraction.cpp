// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;

TEST_F(TransformationTestsF, MatMulConstTransposesExtractionConstantWeights) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 4});
        auto weights = op::v0::Constant::create(element::f32, Shape{1, 3, 2}, {1, 2, 3, 4, 5, 6});
        auto matmul = std::make_shared<op::v0::MatMul>(data, weights, true);
        model = std::make_shared<Model>(matmul, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 4});
        auto weights = op::v0::Constant::create(element::f32, Shape{1, 2, 3}, {1, 3, 5, 2, 4, 6});
        auto matmul = std::make_shared<op::v0::MatMul>(data, weights, true, true);
        model_ref = std::make_shared<Model>(matmul, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, MatMulConstTransposesExtractionFQOnWeights) {
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 3});
        auto weights = op::v0::Constant::create(element::f32, Shape{1, 3, 2}, {1, 2, 3, 4, 5, 6});
        auto low = op::v0::Constant::create(element::f32, Shape{1}, {0});
        auto high = op::v0::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<op::v0::FakeQuantize>(weights, low, high, low, high, 255);
        auto matmul = std::make_shared<op::v0::MatMul>(data, fq);
        model = std::make_shared<Model>(matmul, ParameterVector{data});

        manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 3});
        auto weights = op::v0::Constant::create(element::f32, Shape{1, 3, 2}, {1, 2, 3, 4, 5, 6});
        auto low = op::v0::Constant::create(element::f32, Shape{1}, {0});
        auto high = op::v0::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<op::v0::FakeQuantize>(weights, low, high, low, high, 255);
        auto transpose =
            std::make_shared<op::v1::Transpose>(fq, op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
        auto matmul = std::make_shared<op::v0::MatMul>(data, transpose, false, true);
        model_ref = std::make_shared<Model>(matmul, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, NegativeMatMulConstTransposesExtractionInvalidRank) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 4});
    auto weights = op::v0::Constant::create(element::f32, Shape{3}, {1, 2, 3});
    auto matmul = std::make_shared<op::v0::MatMul>(data, weights, true);
    model = std::make_shared<Model>(matmul, ParameterVector{data});
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NegativeMatMulConstTransposesExtractionTransposeBSet) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 4});
    auto weights = op::v0::Constant::create(element::f32, Shape{1, 2, 3}, {1, 2, 3, 4, 5, 6});
    auto matmul = std::make_shared<op::v0::MatMul>(data, weights, true, true);
    model = std::make_shared<Model>(matmul, ParameterVector{data});
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, NegativeMatMulConstTransposesExtractionNonUnitDims) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 4});
    auto weights = op::v0::Constant::create(element::f32, Shape{2, 3, 2}, {1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7});
    auto matmul = std::make_shared<op::v0::MatMul>(data, weights, true);
    model = std::make_shared<Model>(matmul, ParameterVector{data});
    manager.register_pass<ov::pass::MatMulConstTransposesExtraction>();
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
