// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/convert_prior_box_v8_to_v0.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertPriorBox8To0) {
    {
        const Shape input_shape {2, 2};
        const Shape image_Shape {10, 10};
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;

        auto input = std::make_shared<opset8::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset8::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset8::PriorBox>(input, image, attrs);

        function = std::make_shared<Function>(NodeVector {prior_box}, ParameterVector {input, image});
        manager.register_pass<pass::ConvertPriorBox8To0>();
    }

    {
        const Shape input_shape {2, 2};
        const Shape image_Shape {10, 10};
        op::v0::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;

        auto input = std::make_shared<opset1::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset1::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset1::PriorBox>(input, image, attrs);

        function_ref = std::make_shared<Function>(NodeVector {prior_box}, ParameterVector {input, image});
    }
}

TEST_F(TransformationTestsF, ConvertPriorBox8To0_min_max_aspect_ratios_order) {
    {
        const Shape input_shape {2, 2};
        const Shape image_Shape {10, 10};
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;
        attrs.min_max_aspect_ratios_order = false;

        auto input = std::make_shared<opset8::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset8::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset8::PriorBox>(input, image, attrs);

        function = std::make_shared<Function>(NodeVector {prior_box}, ParameterVector {input, image});
        manager.register_pass<pass::ConvertPriorBox8To0>();
    }
}