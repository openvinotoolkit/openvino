// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace ov;
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
