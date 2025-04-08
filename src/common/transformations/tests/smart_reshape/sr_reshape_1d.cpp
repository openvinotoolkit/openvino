// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"

using namespace ov;

TEST(SmartReshapeTests, Reshape1d) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto reshape = std::make_shared<op::v1::Reshape>(input, op::v0::Constant::create(element::i64, {1}, {5}), true);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({5}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    OV_ASSERT_NO_THROW(f->reshape({{1, 3, 300, 300}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({270000}));
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3, 300, 300}));
}

TEST(SmartReshapeTests, Reshape1d_negative) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto pattern = std::make_shared<op::v0::Parameter>(element::i64, Shape{1});
        auto reshape = std::make_shared<op::v1::Reshape>(input, pattern, false);
        f = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input, pattern});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().is_dynamic());

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{1, 3, 300, 300}}));
}
