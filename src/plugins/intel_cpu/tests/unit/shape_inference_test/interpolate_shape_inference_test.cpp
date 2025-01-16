// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

// --- v0 ---
class InterpolateV0StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v0::Interpolate> {
protected:
    using Attrs = typename op_type::Attributes;

    void SetUp() override {
        output_shapes.resize(1);
    }

    Attrs attrs;
};

TEST_F(InterpolateV0StaticShapeInferenceTest, default_ctor_no_attributes) {
    attrs.axes = AxisSet{2, 0, 5};

    op = make_op();
    op->set_attrs(attrs);

    int32_t out_shape_v[] = {10, 20, 30};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, out_shape_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 2, 20, 128, 128, 30}));
}

TEST_F(InterpolateV0StaticShapeInferenceTest, out_shape_as_constant) {
    attrs.axes = AxisSet{1, 3};

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto out_shape = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{2}, {100, 100});
    op = make_op(img, out_shape, attrs);

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 100, 128, 100, 128}));
}

TEST_F(InterpolateV0StaticShapeInferenceTest, all_inputs_dynamic_rank_use_scales) {
    attrs.axes = AxisSet{2, 4, 5};

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(img, out_shape, attrs);

    int32_t out_shape_v[] = {10, 20, 30};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, out_shape_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 2, 10, 128, 20, 30}));
}

TEST_F(InterpolateV0StaticShapeInferenceTest, all_inputs_static_rank_use_sizes) {
    attrs.axes = AxisSet{0, 1, 2};

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(6));
    const auto out_shape = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    op = make_op(img, out_shape, attrs);

    int32_t out_shape_v[] = {10, 20, 30};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, out_shape_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({10, 20, 30, 128, 128, 64}));
}

// --- v4 ---
class InterpolateV4StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v4::Interpolate> {
protected:
    using Attrs = typename op_type::InterpolateAttrs;
    using ShapeCalcMode = typename op_type::ShapeCalcMode;

    void SetUp() override {
        output_shapes.resize(1);
    }

    Attrs attrs;
};

TEST_F(InterpolateV4StaticShapeInferenceTest, default_ctor_no_attributes) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;

    op = make_op();
    op->set_attrs(attrs);

    float scales_v[] = {1.5f, 3.0f, 0.2f};
    int32_t axes_v[] = {2, 0, 5};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{2, {element::f32, ov::Shape{3}, scales_v}},
                                                                   {3, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({15, 2, 192, 128, 128, 12}));
}

TEST_F(InterpolateV4StaticShapeInferenceTest, scales_as_constant) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    const auto scales = op::v0::Constant::create<float>(element::f32, ov::Shape{2}, {2.0f, 0.7f});
    const auto axes = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{2}, {1, 3});
    op = make_op(img, sizes, scales, axes, attrs);

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128}, {1}, {2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 4, 128, 89, 128}));
}

TEST_F(InterpolateV4StaticShapeInferenceTest, sizes_as_constant) {
    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto sizes = op::v0::Constant::create<float>(element::i32, ov::Shape{2}, {10, 5});
    const auto scales = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    const auto axes = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{2}, {3, 1});
    op = make_op(img, sizes, scales, axes, attrs);

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128}, {2}, {1}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 5, 128, 10, 128}));
}

TEST_F(InterpolateV4StaticShapeInferenceTest, all_inputs_dynamic_rank_use_scales) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.pads_end = std::vector<size_t>(5, 1);

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto scales = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(img, sizes, scales, axes, attrs);

    float scales_v[] = {1.5f, 3.0f, 0.2f};
    int32_t axes_v[] = {2, 0, 5};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{2, {element::f32, ov::Shape{3}, scales_v}},
                                                                   {3, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({18, 3, 193, 129, 129, 12}));
}

TEST_F(InterpolateV4StaticShapeInferenceTest, all_inputs_static_rank_use_sizes) {
    attrs.shape_calculation_mode = ShapeCalcMode::SIZES;

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto scales = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    op = make_op(img, sizes, scales, axes, attrs);

    int32_t sizes_v[] = {10, 50, 60};
    int32_t axes_v[] = {1, 0, 3};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, sizes_v}},
                                                                   {3, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({50, 10, 128, 60, 128, 64}));
}

// --- v11 ---
class InterpolateV11StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v11::Interpolate> {
protected:
    using Attrs = typename op_type::InterpolateAttrs;
    using ShapeCalcMode = typename op_type::ShapeCalcMode;

    void SetUp() override {
        output_shapes.resize(1);
    }

    Attrs attrs;
};

TEST_F(InterpolateV11StaticShapeInferenceTest, default_ctor_no_attributes) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;

    op = make_op();
    op->set_attrs(attrs);

    float scales_v[] = {1.5f, 3.0f, 0.2f};
    int32_t axes_v[] = {2, 0, 5};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::f32, ov::Shape{3}, scales_v}},
                                                                   {2, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({15, 2, 192, 128, 128, 12}));
}

TEST_F(InterpolateV11StaticShapeInferenceTest, scales_as_constant) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scales = op::v0::Constant::create<float>(element::f32, ov::Shape{2}, {2.0f, 0.7f});
    const auto axes = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{2}, {1, 3});
    op = make_op(img, scales, axes, attrs);

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128}, {2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 4, 128, 89, 128}));
}

TEST_F(InterpolateV11StaticShapeInferenceTest, sizes_as_constant) {
    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto sizes = op::v0::Constant::create<float>(element::i32, ov::Shape{2}, {10, 5});
    const auto axes = op::v0::Constant::create<int64_t>(element::i64, ov::Shape{2}, {3, 1});
    op = make_op(img, sizes, axes, attrs);

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128}, {2}, {2}};
    output_shapes = shape_inference(op.get(), input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({5, 5, 128, 10, 128}));
}

TEST_F(InterpolateV11StaticShapeInferenceTest, all_inputs_dynamic_rank_use_scales) {
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto scales = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    op = make_op(img, scales, axes, attrs);

    float scales_v[] = {1.5f, 3.0f, 0.2f};
    int32_t axes_v[] = {2, 0, 5};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::f32, ov::Shape{3}, scales_v}},
                                                                   {2, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({15, 2, 192, 128, 128, 12}));
}

TEST_F(InterpolateV11StaticShapeInferenceTest, all_inputs_static_rank_use_sizes) {
    attrs.shape_calculation_mode = ShapeCalcMode::SIZES;

    const auto img = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic(5));
    const auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    const auto axes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic(1));
    op = make_op(img, sizes, axes, attrs);

    int32_t sizes_v[] = {10, 50, 60};
    int32_t axes_v[] = {1, 0, 3};
    const auto const_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{3}, sizes_v}},
                                                                   {2, {element::i32, ov::Shape{3}, axes_v}}};

    input_shapes = StaticShapeVector{{5, 2, 128, 128, 128, 64}, {3}, {3}};
    const auto output_shapes = shape_inference(op.get(), input_shapes, const_data);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({50, 10, 128, 60, 128, 64}));
}
