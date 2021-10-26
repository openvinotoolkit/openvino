// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct NonMaxSuppressionTestCase {
    int64_t num_batches, num_boxes, num_classes, max_output_boxes_per_class;
    float iou_threshold, score_threshold;
};

using Parameters = std::tuple<
        DataType,
        DataType,
        NonMaxSuppressionTestCase,
        LayerTestsUtils::TargetDevice
>;

class NMS_NonZero : public testing::WithParamInterface<Parameters>,
                    virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& float_type = std::get<0>(parameters);
        const auto& integer_type = std::get<1>(parameters);
        const auto& nms_setup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto boxes = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_boxes, 4});
        const auto scores = std::make_shared<ngraph::opset3::Parameter>(
                float_type, ngraph::PartialShape{nms_setup.num_batches, nms_setup.num_classes, nms_setup.num_boxes});
        const auto max_output_boxes_per_class = std::make_shared<ngraph::opset3::Constant>(
                integer_type, ngraph::Shape{}, std::vector<int64_t>{nms_setup.max_output_boxes_per_class});
        const auto iou_threshold = std::make_shared<ngraph::opset3::Constant>(
                float_type, ngraph::Shape{}, std::vector<float>{nms_setup.iou_threshold});
        const auto score_threshold = std::make_shared<ngraph::opset3::Constant>(
                float_type, ngraph::Shape{}, std::vector<float>{nms_setup.score_threshold});

        const auto NMS = std::make_shared<ngraph::opset3::NonMaxSuppression>(
                boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                ngraph::opset3::NonMaxSuppression::BoxEncodingType::CORNER, false, ngraph::element::i32);

        const auto nonZero = std::make_shared<ngraph::opset3::NonZero>(boxes);

        const auto resultNMS = std::make_shared<ngraph::opset3::Result>(NMS);
        const auto resultNonZero = std::make_shared<ngraph::opset3::Result>(nonZero);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{resultNMS, resultNonZero},
                                                      ngraph::ParameterVector{boxes, scores}, "opset3::NMS-NonZero");
    }
};

TEST_P(NMS_NonZero, CompareWithReference) {
    Run();
}

// #-30919
INSTANTIATE_TEST_SUITE_P(DISABLED_DynamicNonMaxSupression, NMS_NonZero,
                        ::testing::Combine(
                                ::testing::Values(
                                        ngraph::element::f16,
                                        ngraph::element::f32),
                                ::testing::Values(
                                        ngraph::element::i32,
                                        ngraph::element::u8),
                                ::testing::Values(
                                        // num_batches, num_boxes, num_classes, max_output_boxes_per_class, iou_threshold, score_threshold
                                        NonMaxSuppressionTestCase{1, 10, 5, 10, 0., 0.},
                                        NonMaxSuppressionTestCase{2, 100, 5, 10, 0., 0.},
                                        NonMaxSuppressionTestCase{3, 10, 5, 2, 0.5, 0.},
                                        NonMaxSuppressionTestCase{1, 1000, 1, 2000, 0.5, 0.}),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
