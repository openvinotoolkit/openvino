// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;
using ov::op::v0::Parameter;

TEST(type_prop, rms_norm_no_scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, eps);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_scale_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_scale_compute_type_no_scale) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_dynamic_data_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{-1, {3, 4}, {8, -1}, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {3, 4}, {8, -1}, 6}));
}

TEST(type_prop, rms_norm_dynamic_data_shape_rank) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, rms_norm_incorrect_input_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;
    {
        const auto data_int = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data_int, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the data tensor must be a floating point type"));
    }
    {
        const auto axes_float = std::make_shared<Parameter>(element::f32, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes_float, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the axes tensor must be i32 or i64 type"));
    }
}

TEST(type_prop, rms_norm_incompatible_axes_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5f;
    const auto compute_type = element::f32;
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Expected 1D tensor for the 'axes' input"));
    }
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{4});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Number of the axes can't be higher than the rank of the data shape"));
    }
}
