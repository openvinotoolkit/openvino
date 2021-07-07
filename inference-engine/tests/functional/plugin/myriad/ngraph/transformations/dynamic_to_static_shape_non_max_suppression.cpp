// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/op/non_max_suppression.hpp>
#include <ngraph/output_vector.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp>
#include <vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp>

#include <vpu/utils/error.hpp>

#include <numeric>
#include <queue>
#include <random>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct NonMaxSuppressionTestCase {
    int64_t numBatches, numBoxes, numClasses, maxOutputBoxesPerClass;
    float iouThreshold, scoreThreshold, softNMSSigma;
};

class DynamicToStaticShapeNonMaxSuppression : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, NonMaxSuppressionTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& floatType = std::get<0>(parameters);
        const auto& integerType = std::get<1>(parameters);
        const auto& nmsSetup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(floatType, integerType, nmsSetup),
                *reference(floatType, integerType, nmsSetup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& floatType,
            const ngraph::element::Type_t& integerType,
            const NonMaxSuppressionTestCase& nmsSetup) const {
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(
                floatType, ngraph::PartialShape{nmsSetup.numBatches, nmsSetup.numBoxes, 4});
        const auto scores = std::make_shared<ngraph::opset3::Parameter>(
                floatType, ngraph::PartialShape{nmsSetup.numBatches, nmsSetup.numClasses, nmsSetup.numBoxes});

        const auto maxOutputBoxesPerClass = ngraph::opset3::Constant::create(integerType, {}, std::vector<int64_t>{nmsSetup.maxOutputBoxesPerClass});
        const auto iouThreshold = ngraph::opset3::Constant::create(floatType, ngraph::Shape{}, std::vector<float>{nmsSetup.iouThreshold});
        const auto scoreThreshold = ngraph::opset3::Constant::create(floatType, ngraph::Shape{}, std::vector<float>{nmsSetup.scoreThreshold});
        const auto softNMSSigma = ngraph::opset3::Constant::create(floatType, ngraph::Shape{}, std::vector<float>{nmsSetup.softNMSSigma});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{3});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(scores, dims);

        const auto node = std::make_shared<ngraph::op::v5::NonMaxSuppression>(
                boxes, dsr, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, softNMSSigma);

        const auto function = std::make_shared<ngraph::Function>(
                node->outputs(),
                ngraph::ParameterVector{boxes, scores, dims},
                "Actual");

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticNonMaxSuppression}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& floatType,
            const ngraph::element::Type_t& integerType,
            const NonMaxSuppressionTestCase& nmsSetup) const {
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(
                floatType, ngraph::PartialShape{nmsSetup.numBatches, nmsSetup.numBoxes, 4});
        const auto scores = std::make_shared<ngraph::opset3::Parameter>(
                floatType, ngraph::PartialShape{nmsSetup.numBatches, nmsSetup.numClasses, nmsSetup.numBoxes});

        const auto maxOutputBoxesPerClass = ngraph::opset3::Constant::create(integerType, {}, std::vector<int64_t>{nmsSetup.maxOutputBoxesPerClass});
        const auto iouThreshold = ngraph::opset3::Constant::create(floatType, {}, std::vector<float>{nmsSetup.iouThreshold});
        const auto scoreThreshold = ngraph::opset3::Constant::create(floatType, {}, std::vector<float>{nmsSetup.scoreThreshold});
        const auto softNMSSigma = ngraph::opset3::Constant::create(floatType, ngraph::Shape{}, std::vector<float>{nmsSetup.softNMSSigma});

        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{3});
        const auto dsrInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(scores, dims);

        const auto node = std::make_shared<ngraph::vpu::op::StaticShapeNonMaxSuppression>(
                boxes, dsrInput, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, softNMSSigma);

        const auto validOutputs = std::make_shared<ngraph::opset5::Gather>(
                node->output(2),
                ngraph::opset5::Constant::create(dims->get_element_type(), ngraph::Shape{1}, {0}),
                ngraph::opset5::Constant::create(dims->get_element_type(), ngraph::Shape{1}, {0}));

        const auto dsrIndices = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), node->output(2));
        const auto dsrScores = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(1), node->output(2));

        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsrIndices, dsrScores, validOutputs},
                ngraph::ParameterVector{boxes, scores, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeNonMaxSuppression, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeNonMaxSuppression, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32),
    testing::Values(
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        // numBatches, numBoxes, numClasses, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, softNMSSigma
        NonMaxSuppressionTestCase{1, 10, 5, 10, 0., 0., 0.},
        NonMaxSuppressionTestCase{2, 100, 5, 10, 0., 0., 0.},
        NonMaxSuppressionTestCase{3, 10, 5, 2, 0.5, 0., 0.},
        NonMaxSuppressionTestCase{1, 1000, 1, 2000, 0.5, 0., 0.})));

}  // namespace
