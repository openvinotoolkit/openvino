// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_multiclass_nms_upgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9_dynamic_rank) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertMulticlassNms8ToMulticlassNms9_dynamic_dims) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32,
                                                         PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
        auto scores = std::make_shared<opset1::Parameter>(
            element::f32,
            PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));

        auto nms = std::make_shared<opset8::MulticlassNms>(boxes, scores, opset8::MulticlassNms::Attributes());

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertMulticlassNms8ToMulticlassNms9>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32,
                                                         PartialShape({Dimension::dynamic(), Dimension::dynamic(), 4}));
        auto scores = std::make_shared<opset1::Parameter>(
            element::f32,
            PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}
