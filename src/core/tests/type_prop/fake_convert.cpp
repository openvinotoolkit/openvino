// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using ov::op::v0::Parameter;

TEST(type_prop, fake_convert_no_shift) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

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

TEST(type_prop, fake_convert_basic_bf16) {
    const auto data = std::make_shared<Parameter>(element::bf16, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::bf16, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::bf16, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, fake_convert_basic_dynamic) {
    const auto data = std::make_shared<Parameter>(element::dynamic, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::dynamic, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::dynamic, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
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

TEST(type_prop, fake_convert_symbol) {
    PartialShape data_shape{2, 1, Dimension::dynamic(), 6};
    auto symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, Dimension::dynamic(), 6}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, fake_convert_example_0) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 1, 3, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 6}));
}

TEST(type_prop, fake_convert_example_1) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 1, 3, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{3, 1});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 6}));
}
TEST(type_prop, fake_convert_example_2) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 1, {3, 5}, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{3, 1});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 6}));
}
TEST(type_prop, fake_convert_example_3) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, {2, 5}, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 1});

    const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 3, 6}));
}

TEST(type_prop, fake_convert_basic_unsupported_type) {
    const auto data = std::make_shared<Parameter>(element::f64, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f64, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f64, PartialShape{});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift),
                    Exception,
                    testing::HasSubstr("The element type of the input tensor must be a bf16, f16, f32 but got: f64"));
}

TEST(type_prop, fake_convert_basic_unsupported_shape_scale_shift) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift),
                    AssertFailure,
                    testing::HasSubstr("FakeConvert scale shape is not compatible with shift shape."));
}

TEST(type_prop, fake_convert_basic_unsupported_shape_not_broadcastable) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift),
                    AssertFailure,
                    testing::HasSubstr("Argument shapes are inconsistent."));
}

TEST(type_prop, fake_convert_basic_unsupported_shape_not_unidirectional_broadcastable) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 1, 6});
    const auto scale = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});
    const auto shift = std::make_shared<Parameter>(element::f32, PartialShape{1, 1, 3, 1});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift),
                    AssertFailure,
                    testing::HasSubstr(
                        "FakeConvert support only unidirectional broadcasting, inputs cannot be broadcast into data."));
}
TEST(type_prop, fake_convert_basic_unsupported_mixed_types) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(element::dynamic, PartialShape{});
    const auto shift = std::make_shared<Parameter>(element::f16, PartialShape{});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v13::FakeConvert>(data, scale, shift),
                    AssertFailure,
                    testing::HasSubstr("Mixed input types are not supported."));
}
