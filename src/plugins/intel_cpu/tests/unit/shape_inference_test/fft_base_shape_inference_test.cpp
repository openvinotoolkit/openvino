// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;

static std::shared_ptr<op::v7::DFT> build_dft() {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto DFT = std::make_shared<ov::op::v7::DFT>(input_shape, axes);
    return DFT;
}

static std::shared_ptr<op::v7::DFT> build_dft_signal() {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto DFT_signal = std::make_shared<ov::op::v7::DFT>(input_shape, axes, signal);
    return DFT_signal;
}

static std::shared_ptr<op::v7::DFT> build_dft_constant() {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{1, 2});
    auto signal = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{512, 100});
    auto DFT_signal = std::make_shared<ov::op::v7::DFT>(input_shape, axes, signal);
    return DFT_signal;
}

static std::shared_ptr<op::v7::IDFT> build_idft() {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto IDFT = std::make_shared<ov::op::v7::IDFT>(input_shape, axes);
    return IDFT;
}

static std::shared_ptr<op::v7::IDFT> build_idft_signal() {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto IDFT_signal = std::make_shared<ov::op::v7::IDFT>(input_shape, axes, signal);
    return IDFT_signal;
}

TEST(StaticShapeInferenceTest, DFTTest) {
    auto DFT = build_dft();
    int32_t axes_val[] = {1, 2};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(DFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 320, 320, 2}));
}

TEST(StaticShapeInferenceTest, DFTSignalTest) {
    auto DFT = build_dft_signal();

    int32_t axes_val[] = {1, 2};
    int32_t signal_val[] = {512, 100};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}},
                                                                {2, {element::i32, ov::Shape{2}, signal_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(DFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));
}

TEST(StaticShapeInferenceTest, DFTConstantTest) {
    auto DFT = build_dft_constant();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(DFT.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));
}

TEST(StaticShapeInferenceTest, DFTSignalMissingConstDataTest) {
    auto DFT = build_dft_signal();

    int32_t axes_val[] = {1, 2};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};

    EXPECT_THROW(shape_inference(DFT.get(), static_input_shapes, constant_data), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, IDFTTest) {
    auto IDFT = build_idft();
    int32_t axes_val[] = {1, 2};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(IDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 320, 320, 2}));
}

TEST(StaticShapeInferenceTest, IDFTSignalTest) {
    auto IDFT = build_idft_signal();

    int32_t axes_val[] = {1, 2};
    int32_t signal_val[] = {512, 100};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}},
                                                                {2, {element::i32, ov::Shape{2}, signal_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(IDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));
}

TEST(StaticShapeInferenceTest, IDFTSignalMissingConstDataTest) {
    auto IDFT = build_idft_signal();

    int32_t axes_val[] = {1, 2};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(IDFT.get(), static_input_shapes, constant_data), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, RDFT) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto RDFT = std::make_shared<ov::op::v9::RDFT>(input_shape, axes);

    int32_t axes_val[] = {2, 3};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 64}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(RDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 120, 64, 33, 2}));
}

TEST(StaticShapeInferenceTest, RDFTWithSignalSizes) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto RDFT = std::make_shared<ov::op::v9::RDFT>(input_shape, axes, signal);

    int32_t axes_val[] = {2, 3};
    int32_t signal_val[] = {40, 30};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}},
                                                                {2, {element::i32, ov::Shape{2}, signal_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 64}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(RDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 120, 40, 16, 2}));
}

TEST(StaticShapeInferenceTest, RDFTWithConstAxesAndSignalSizes) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{2, 3});
    auto signal = std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape{2}, std::vector<int32_t>{64, 64});
    auto RDFT = std::make_shared<ov::op::v9::RDFT>(input_shape, axes, signal);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 64}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(RDFT.get(), static_input_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 120, 64, 33, 2}));
}

TEST(StaticShapeInferenceTest, RDFTMissingSignalTensor) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto RDFT = std::make_shared<ov::op::v9::RDFT>(input_shape, axes, signal);

    int32_t axes_val[] = {2, 3};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 64}, StaticShape{2}, StaticShape{2}};
    EXPECT_THROW(shape_inference(RDFT.get(), static_input_shapes, constant_data), NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, IRDFT) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto IRDFT = std::make_shared<ov::op::v9::IRDFT>(input_shape, axes);

    int32_t axes_val[] = {2, 3};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 33, 2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(IRDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 120, 64, 64}));
}

TEST(StaticShapeInferenceTest, IRDFTWithSignalSizes) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto IRDFT = std::make_shared<ov::op::v9::IRDFT>(input_shape, axes, signal);

    int32_t axes_val[] = {2, 3};
    int32_t signal_val[] = {64, 64};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}},
                                                                {2, {element::i32, ov::Shape{2}, signal_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 33, 2}, StaticShape{2}, StaticShape{2}};

    const auto static_output_shapes = shape_inference(IRDFT.get(), static_input_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 120, 64, 64}));
}

TEST(StaticShapeInferenceTest, IRDFTMissingSignalSizesTensor) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1, -1});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto signal = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto IRDFT = std::make_shared<ov::op::v9::IRDFT>(input_shape, axes, signal);

    int32_t axes_val[] = {2, 3};
    auto constant_data = std::unordered_map<size_t, ov::Tensor>{{1, {element::i32, ov::Shape{2}, axes_val}}};

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 120, 64, 33, 2}, StaticShape{2}, StaticShape{2}};
    EXPECT_THROW(shape_inference(IRDFT.get(), static_input_shapes, constant_data), NodeValidationFailure);
}
