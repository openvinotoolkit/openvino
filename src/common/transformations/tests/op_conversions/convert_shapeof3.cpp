// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_shapeof3.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, ConvertShapeOf3WithI64) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        auto shapeof = std::make_shared<opset3::ShapeOf>(input, element::i64);
        shapeof->set_friendly_name("shapeof");

        model = std::make_shared<ov::Model>(NodeVector{shapeof}, ParameterVector{input});

        manager.register_pass<ov::pass::ConvertShapeOf3>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        auto shapeof = std::make_shared<opset1::ShapeOf>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{shapeof}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertShapeOf3WithI32) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        auto shapeof = std::make_shared<opset3::ShapeOf>(input, element::i32);

        model = std::make_shared<ov::Model>(NodeVector{shapeof}, ParameterVector{input});

        manager.register_pass<ov::pass::ConvertShapeOf3>();
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 2, 3});
        auto shapeof = std::make_shared<opset1::ShapeOf>(input);
        auto convert = std::make_shared<opset1::Convert>(shapeof, element::i32);

        model_ref = std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{input});
    }
}
