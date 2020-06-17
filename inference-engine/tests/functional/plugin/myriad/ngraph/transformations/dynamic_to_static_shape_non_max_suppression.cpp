// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset3.hpp>
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
    int64_t num_batches, num_boxes, num_classes, max_output_boxes_per_class;
    float iou_threshold, score_threshold;
};


class DynamicToStaticShapeNonMaxSuppression : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<std::tuple<DataType, DataType, NonMaxSuppressionTestCase>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& nms_setup = std::get<2>(parameters);

        ngraph::helpers::CompareFunctions(*transform(float_type, integer_type, nms_setup),
                *reference(float_type, integer_type, nms_setup));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const NonMaxSuppressionTestCase& nms_setup) const {
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_boxes, 4});
        const auto scores = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_classes, nms_setup.num_boxes});
        const auto max_output_boxes_per_class = ngraph::opset3::Constant::create(integer_type, {}, std::vector<int64_t>{nms_setup.max_output_boxes_per_class});
        const auto iou_threshold = ngraph::opset3::Constant::create(float_type, ngraph::Shape{}, std::vector<float>{nms_setup.iou_threshold});
        const auto score_threshold = ngraph::opset3::Constant::create(float_type, ngraph::Shape{}, std::vector<float>{nms_setup.score_threshold});


        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{3});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(scores, dims);

        const auto node = std::make_shared<ngraph::op::v4::NonMaxSuppression>(
                boxes, dsr, max_output_boxes_per_class, iou_threshold, score_threshold);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                ngraph::ParameterVector{boxes, scores, dims},
                "Actual");
        node->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(outputShape.rank()));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticNonMaxSuppression}};
        vpu::DynamicToStaticShape(transformations).transform(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& float_type,
            const ngraph::element::Type_t& integer_type,
            const NonMaxSuppressionTestCase& nms_setup) const {
        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_boxes, 4});
        const auto scores = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_classes, nms_setup.num_boxes});
        const auto max_output_boxes_per_class = ngraph::opset3::Constant::create(integer_type, {}, std::vector<int64_t>{nms_setup.max_output_boxes_per_class});
        const auto iou_threshold = ngraph::opset3::Constant::create(float_type, {}, std::vector<float>{nms_setup.iou_threshold});
        const auto score_threshold = ngraph::opset3::Constant::create(float_type, {}, std::vector<float>{nms_setup.score_threshold});


        const auto dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{3});
        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(scores, dims);

        const auto node = std::make_shared<ngraph::vpu::op::StaticShapeNonMaxSuppression>(
                boxes, dsr, max_output_boxes_per_class, iou_threshold, score_threshold);

         const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node->output(0), node->output(1));
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{dsr1},
                ngraph::ParameterVector{boxes, scores, dims},
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeNonMaxSuppression, CompareFunctions) {
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicToStaticShapeNonMaxSuppression, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32),
    testing::Values(
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        // num_batches, num_boxes, num_classes, max_output_boxes_per_class, iou_threshold, score_threshold
        NonMaxSuppressionTestCase{1, 10, 5, 10, 0., 0.},
        NonMaxSuppressionTestCase{2, 100, 5, 10, 0., 0.},
        NonMaxSuppressionTestCase{3, 10, 5, 2, 0.5, 0.},
        NonMaxSuppressionTestCase{1, 1000, 1, 2000, 0.5, 0.})));

}  // namespace
