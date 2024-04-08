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

TEST(type_prop, rms_norm_default_ctor) {
    const auto op = std::make_shared<op::v14::RMSNorm>();
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i64, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});

    op->set_arguments(ov::OutputVector{data, axes, scale});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_no_scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, eps);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
}

TEST(type_prop, rms_norm_scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
}

TEST(type_prop, rms_norm_scale_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
    EXPECT_EQ(op->get_compute_type(), compute_type);
}

TEST(type_prop, rms_norm_scale_compute_type_no_scale) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, eps, compute_type);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST(type_prop, rms_norm_dynamic_data_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{-1, {3, 4}, {8, -1}, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {3, 4}, {8, -1}, 6}));
}

TEST(type_prop, rms_norm_dynamic_data_shape_rank) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST(type_prop, rms_norm_propagate_symbols) {
    auto data_shape = PartialShape{-1, {3, 4}, {8, -1}, 6};
    set_shape_symbols(data_shape);
    const auto exp_symbols = get_shape_symbols(data_shape);

    const auto data = std::make_shared<Parameter>(element::f16, data_shape);
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;

    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), exp_symbols);
}

TEST(type_prop, rms_norm_incorrect_input_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
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
    {
        const auto scale_incompatible = std::make_shared<Parameter>(element::f32, PartialShape{1});
        OV_EXPECT_THROW(
            std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale_incompatible, eps, compute_type),
            ov::NodeValidationFailure,
            HasSubstr("Element type of the scale input must be the same as the data input type"));
    }
}

TEST(type_prop, rms_norm_incompatible_axes_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Axes input must be a scalar or 1D input. Got: [1,2]"));
    }
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{4});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Number of the axes can't be higher than the rank of the data shape"));
    }
}

using RMSNormTestParam = std::tuple<PartialShape, PartialShape>;
class RMSNormTest : public TypePropOpTest<ov::op::v14::RMSNorm>, public WithParamInterface<RMSNormTestParam> {
protected:
    void SetUp() override {
        std::tie(shape_data, shape_scale) = GetParam();
    }
    PartialShape shape_data, shape_scale;
};

INSTANTIATE_TEST_SUITE_P(type_prop_rms_scale_shape,
                         RMSNormTest,
                         Values(std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{-1}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{1}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{2}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{1, 1}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{1, 2}),
                                std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{3, 1, 2}),
                                std::make_tuple(PartialShape{-1, 4, 8, 6}, PartialShape{1, 4, 1, 1}),
                                std::make_tuple(PartialShape{2, 4, 8, 6}, PartialShape{2, 4, 8, 6}),
                                std::make_tuple(PartialShape{2, 4, 8, 6}, PartialShape{1, 4, 1, 1}),
                                std::make_tuple(PartialShape{2, 4, 8, 6}, PartialShape{1, 1, 1, 1}),
                                std::make_tuple(PartialShape{2, 4, 8, 6}, PartialShape::dynamic()),
                                std::make_tuple(PartialShape::dynamic(), PartialShape{1}),
                                std::make_tuple(PartialShape::dynamic(), PartialShape::dynamic())),
                         PrintToStringParamName());

TEST_P(RMSNormTest, scale_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, shape_data);
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5;

    const auto scale = std::make_shared<Parameter>(element::f16, shape_scale);
    const auto op = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps);

    EXPECT_EQ(op->get_output_partial_shape(0), shape_data);
}

TEST(type_prop, rms_norm_scale_incompatible_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{-1, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto eps = 1e-5;
    const auto compute_type = element::f32;
    {
        const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{8});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Scale input shape must be broadcastable to the shape of the data input"));
    }
    {
        const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{6, 1});
        OV_EXPECT_THROW(std::ignore = std::make_shared<op::v14::RMSNorm>(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Scale input shape must be broadcastable to the shape of the data input"));
    }
}
