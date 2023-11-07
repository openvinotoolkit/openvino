// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace ov;
using ov::op::v0::Parameter;

TEST(type_prop, fake_convert_basic_f32) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, fake_convert_basic_f16) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f16, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, fake_convert_dynamic_shape) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}
