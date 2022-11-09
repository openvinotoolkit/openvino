// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "transformations/smart_reshape/shape_of_const_folding.hpp"


using namespace ov;

TEST_F(TransformationTestsF, ShapeOfConstFolding) {
    {
        auto input = std::make_shared<opset10::Constant>(element::f32, Shape{10, 20}, 1);
        auto shape_of = std::make_shared<opset10::ShapeOf>(input);
        function = std::make_shared<ngraph::Function>(shape_of, ParameterVector{});
    }

    manager.register_pass<ov::pass::ShapeOfConstFolding>();

    {
        auto input = std::make_shared<opset10::Constant>(element::i64, Shape{2}, std::vector<int64_t>{10, 20});
        function_ref = std::make_shared<ngraph::Function>(input, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, ShapeOfConstFoldingNegative) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{10, 20});
        auto shape_of = std::make_shared<opset10::ShapeOf>(input);
        function = std::make_shared<ngraph::Function>(shape_of, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ShapeOfConstFolding>();
}
