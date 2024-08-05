// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class TypePropRMSNormTest : public TypePropOpTest<op::internal::RMSNorm> {
public:
    double eps = 1e-5;
};

TEST_F(TypePropRMSNormTest, default_ctor) {
    const auto op = make_op();
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

TEST_F(TypePropRMSNormTest, no_scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});

    const auto op = make_op(data, axes, eps);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
}

TEST_F(TypePropRMSNormTest, scale_no_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});

    const auto op = make_op(data, axes, scale, eps);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
}

TEST_F(TypePropRMSNormTest, scale_compute_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;

    const auto op = make_op(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
    EXPECT_EQ(op->get_epsilon(), eps);
    EXPECT_EQ(op->get_compute_type(), compute_type);
}

TEST_F(TypePropRMSNormTest, scale_compute_type_no_scale) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto compute_type = element::f32;

    const auto op = make_op(data, axes, eps, compute_type);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8, 6}));
}

TEST_F(TypePropRMSNormTest, dynamic_data_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{-1, {3, 4}, {8, -1}, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;

    const auto op = make_op(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {3, 4}, {8, -1}, 6}));
}

TEST_F(TypePropRMSNormTest, dynamic_data_shape_rank) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;

    const auto op = make_op(data, axes, scale, eps, compute_type);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST_F(TypePropRMSNormTest, propagate_symbols) {
    auto data_shape = PartialShape{-1, {3, 4}, {8, -1}, 6};
    set_shape_symbols(data_shape);
    const auto exp_symbols = get_shape_symbols(data_shape);

    const auto data = std::make_shared<Parameter>(element::f16, data_shape);
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;

    const auto op = make_op(data, axes, scale, eps, compute_type);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), exp_symbols);
}

TEST_F(TypePropRMSNormTest, incorrect_input_type) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;
    {
        const auto data_int = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
        OV_EXPECT_THROW(std::ignore = make_op(data_int, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the data tensor must be a floating point type"));
    }
    {
        const auto axes_float = std::make_shared<Parameter>(element::f32, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes_float, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the axes tensor must be i32 or i64 type"));
    }
    {
        const auto scale_incompatible = std::make_shared<Parameter>(element::f32, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, scale_incompatible, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Element type of the scale input must be the same as the data input type"));
    }
}

TEST_F(TypePropRMSNormTest, incompatible_axes_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{});
    const auto compute_type = element::f32;
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1, 2});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Axes input must be a scalar or 1D input. Got: [1,2]"));
    }
    {
        const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{4});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Number of the axes can't be higher than the rank of the data shape"));
    }
}

TEST_F(TypePropRMSNormTest, constant_axes_val_data_dyn_rank) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto axes = std::make_shared<Constant>(element::i32, Shape{}, 1);
    const auto op = make_op(data, axes, eps);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
}

TEST_F(TypePropRMSNormTest, constant_axes_val_data_static_rank) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    const auto axes = std::make_shared<Constant>(element::i32, Shape{}, 1);
    const auto op = make_op(data, axes, eps);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8}));
}

TEST_F(TypePropRMSNormTest, axes_val_as_shape_of) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    const auto data_rank = std::make_shared<op::v3::ShapeOf>(std::make_shared<op::v3::ShapeOf>(data));
    const auto axes =
        std::make_shared<op::v1::Subtract>(data_rank, std::make_shared<Constant>(element::i64, Shape{}, 1));
    const auto op = make_op(data, axes, eps);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 3, 8}));
}

TEST_F(TypePropRMSNormTest, incorrect_axes_val) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{2, 3, 8});
    {
        const auto axes = std::make_shared<Constant>(element::i32, Shape{}, 3);
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, eps),
                        ov::NodeValidationFailure,
                        HasSubstr("Axis 3 out of the tensor rank range [-3, 2]"));
    }
    {
        const auto axes = std::make_shared<Constant>(element::i32, Shape{}, -4);
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, eps),
                        ov::NodeValidationFailure,
                        HasSubstr("Axis -4 out of the tensor rank range [-3, 2]"));
    }
}

using RMSNormTestParam = std::tuple<PartialShape, PartialShape>;
class TypePropRMSNormTestP : public TypePropRMSNormTest, public testing::WithParamInterface<RMSNormTestParam> {
protected:
    void SetUp() override {
        std::tie(shape_data, shape_scale) = GetParam();
    }
    PartialShape shape_data, shape_scale;
};

INSTANTIATE_TEST_SUITE_P(type_prop_rms_scale_shape,
                         TypePropRMSNormTestP,
                         testing::Values(std::make_tuple(PartialShape{-1, 3, 1, 2}, PartialShape{-1}),
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
                         testing::PrintToStringParamName());

TEST_P(TypePropRMSNormTestP, scale_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, shape_data);
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});

    const auto scale = std::make_shared<Parameter>(element::f16, shape_scale);
    const auto op = make_op(data, axes, scale, eps);

    EXPECT_EQ(op->get_output_partial_shape(0), shape_data);
}

TEST_F(TypePropRMSNormTest, scale_incompatible_shape) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape{-1, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto compute_type = element::f32;
    {
        const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{8});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Scale input shape must be broadcastable to the shape of the data input"));
    }
    {
        const auto scale = std::make_shared<Parameter>(element::f16, PartialShape{6, 1});
        OV_EXPECT_THROW(std::ignore = make_op(data, axes, scale, eps, compute_type),
                        ov::NodeValidationFailure,
                        HasSubstr("Scale input shape must be broadcastable to the shape of the data input"));
    }
}

}  // namespace test
}  // namespace ov
