// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEStaticSixInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               soft_nms_sigma, opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto new_soft_nms_sigma = std::make_shared<opset5::Reshape>(soft_nms_sigma,
                                                                    opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                             one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              new_soft_nms_sigma, 0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEStaticFiveInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEStaticFourInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEStaticThreeInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEStaticTwoInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i32, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic1SixInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               soft_nms_sigma, opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto new_soft_nms_sigma = std::make_shared<opset5::Reshape>(soft_nms_sigma,
                                                                    opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                             one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              new_soft_nms_sigma, 0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic1FiveInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic1FourInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic1ThreeInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic1TwoInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset5::Constant::create(element::i32, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic2SixInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               soft_nms_sigma, opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = ngraph::opset5::Constant::create(element::f32, Shape{}, {0.25});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto new_soft_nms_sigma = std::make_shared<opset5::Reshape>(soft_nms_sigma,
                                                                    opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                             one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              new_soft_nms_sigma, 0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic2FiveInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic2FourInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic2ThreeInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5ToNMSIEDynamic2TwoInputs) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<pass::ConvertNMS5ToLegacyMatcher>();
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i32, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});

        auto one_dim_shape = Shape{1};
        auto new_max_per_class = std::make_shared<opset5::Reshape>(max_output_boxes_per_class,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_iou_threshold = std::make_shared<opset5::Reshape>(iou_threshold,
                                                                   opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        auto new_score_threshold = std::make_shared<opset5::Reshape>(score_threshold,
                                                                     opset5::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                              one_dim_shape), true);
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, new_max_per_class, new_iou_threshold, new_score_threshold,
                                                              0, true, element::i32);

        auto convert_0 = std::make_shared<opset1::Convert>(nms->output(0), element::i64);

        function_ref = std::make_shared<Function>(NodeVector{convert_0}, ParameterVector{boxes, scores});
    }
}
