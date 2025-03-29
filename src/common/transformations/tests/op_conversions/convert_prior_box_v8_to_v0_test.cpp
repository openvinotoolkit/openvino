// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertPriorBox8To0) {
    {
        const Shape input_shape{2};
        const Shape image_Shape{2};
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;

        auto input = std::make_shared<opset8::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset8::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset8::PriorBox>(input, image, attrs);

        model = std::make_shared<Model>(NodeVector{prior_box}, ParameterVector{input, image});
        manager.register_pass<ov::pass::ConvertPriorBox8To0>();
    }

    {
        const Shape input_shape{2};
        const Shape image_Shape{2};
        op::v0::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;

        auto input = std::make_shared<opset1::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset1::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset1::PriorBox>(input, image, attrs);

        model_ref = std::make_shared<Model>(NodeVector{prior_box}, ParameterVector{input, image});
    }
}

TEST_F(TransformationTestsF, ConvertPriorBox8To0_min_max_aspect_ratios_order) {
    {
        const Shape input_shape{2};
        const Shape image_Shape{2};
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;
        attrs.min_max_aspect_ratios_order = false;

        auto input = std::make_shared<opset8::Parameter>(element::i64, input_shape);
        auto image = std::make_shared<opset8::Parameter>(element::i64, image_Shape);

        auto prior_box = std::make_shared<opset8::PriorBox>(input, image, attrs);

        model = std::make_shared<Model>(NodeVector{prior_box}, ParameterVector{input, image});
        manager.register_pass<ov::pass::ConvertPriorBox8To0>();
    }
}
