// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_multiclass_nms_upgrade.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9_dynamic_rank) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9_dynamic_dims) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}