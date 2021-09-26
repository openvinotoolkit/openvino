// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/eliminate_shapeof_after_dsr.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>

#include <common_test_utils/test_common.hpp>
#include <gtest/gtest.h>
#include <ngraph/pass/manager.hpp>


namespace {

using TensorType  = ngraph::element::Type_t;
using TensorShape = ngraph::Shape;

class EliminateShapeOfAfterDSRTest : public CommonTestUtils::TestsCommon,
                                     public testing::WithParamInterface<std::tuple<TensorType, TensorShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type  = std::get<0>(parameters);
        const auto& data_shape = std::get<1>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, data_shape),
                                          *reference(data_type, data_shape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(dsr->output(0));

        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{shapeOf},
                ngraph::ParameterVector{data, shape},
                "Actual");
        ngraph::pass::Manager manager;
        manager.register_pass<vpu::EliminateShapeOfAfterDSR>();
        manager.run_passes(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{shape},
                ngraph::ParameterVector{data, shape},
                "Expected");
    }
};

TEST_P(EliminateShapeOfAfterDSRTest, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, EliminateShapeOfAfterDSRTest, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
                TensorShape{1000},
                TensorShape{4, 1000},
                TensorShape{3, 128, 256},
                TensorShape{2, 3, 128, 256})
));

class EliminateShapeOfAfterDSRWithoutOutputDSR : public CommonTestUtils::TestsCommon,
                                                 public testing::WithParamInterface<std::tuple<TensorType, TensorShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type  = std::get<0>(parameters);
        const auto& data_shape = std::get<1>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, data_shape),
                                          *reference(data_type, data_shape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(dsr->output(0));
        const auto shapeOfOutputRelu = std::make_shared<ngraph::opset3::Relu>(shapeOf->output(0));

        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{shapeOfOutputRelu},
                ngraph::ParameterVector{data, shape},
                "Actual");
        ngraph::pass::Manager manager;
        manager.register_pass<vpu::EliminateShapeOfAfterDSR>();
        manager.run_passes(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        const auto shapeRelu = std::make_shared<ngraph::opset3::Relu>(shape);

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{shapeRelu},
                ngraph::ParameterVector{data, shape},
                "Expected");
    }
};

TEST_P(EliminateShapeOfAfterDSRWithoutOutputDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, EliminateShapeOfAfterDSRWithoutOutputDSR, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
                TensorShape{1000},
                TensorShape{4, 1000},
                TensorShape{3, 128, 256},
                TensorShape{2, 3, 128, 256})
));

class EliminateShapeOfAfterDSRKeepDSR : public CommonTestUtils::TestsCommon,
                                        public testing::WithParamInterface<std::tuple<TensorType, TensorShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& data_type  = std::get<0>(parameters);
        const auto& data_shape = std::get<1>(parameters);

        ngraph::helpers::CompareFunctions(*transform(data_type, data_shape),
                                          *reference(data_type, data_shape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(dsr->output(0));
        const auto dsrOutputRelu = std::make_shared<ngraph::opset3::Relu>(dsr->output(0));
        const auto shapeOfOutputRelu = std::make_shared<ngraph::opset3::Relu>(shapeOf->output(0));

        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsrOutputRelu},
                ngraph::ParameterVector{data, shape},
                "Actual");

        ngraph::pass::Manager manager;
        manager.register_pass<vpu::EliminateShapeOfAfterDSR>();
        manager.run_passes(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& dataType,
            const TensorShape& dataShape) const {
        const auto data = std::make_shared<ngraph::opset3::Parameter>(dataType, dataShape);
        const auto shape = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataShape.size()});

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, shape);
        const auto shapeRelu = std::make_shared<ngraph::opset3::Relu>(shape);
        const auto dsrOutputRelu = std::make_shared<ngraph::opset3::Relu>(dsr->output(0));

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsrOutputRelu},
                ngraph::ParameterVector{data, shape},
                "Expected");
    }
};

TEST_P(EliminateShapeOfAfterDSRKeepDSR, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, EliminateShapeOfAfterDSRKeepDSR, testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64,
                ngraph::element::u8),
        testing::Values(
                TensorShape{1000},
                TensorShape{4, 1000},
                TensorShape{3, 128, 256},
                TensorShape{2, 3, 128, 256})
));

}  // namespace
