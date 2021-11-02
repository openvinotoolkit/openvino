// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fft_base_shape_inference.hpp>
#include <openvino/op/dft.hpp>
#include <openvino/op/idft.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, DFTTest) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes_shape = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});
    auto signal = op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {512, 100});
    auto DFT = std::make_shared<ov::op::v7::DFT>(input_shape, axes_shape);
    auto DFT_signal = std::make_shared<ov::op::v7::DFT>(input_shape, axes_shape, signal);

    // DFT StaticShape
    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_infer(DFT.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 320, 320, 2}));

    // DFT PartialShape
    std::vector<PartialShape> input_shapes = {PartialShape{1, 320, 320, 2}, PartialShape{2}},
                              output_shapes = {PartialShape{}};
    shape_infer(DFT.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, 320, 320, 2}));

    // DFT StaticShape with signal
    static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    shape_infer(DFT_signal.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));

    // DFT PartialShape with signal
    input_shapes = {PartialShape{1, 320, 320, 2}, PartialShape{2}, PartialShape{2}};
    output_shapes = {PartialShape{}};
    shape_infer(DFT_signal.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, 512, 100, 2}));

    // DFT StaticShape with one constData
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constData;
    int32_t ptr[2] = {1, 2};
    constData[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, ptr);
    static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    shape_infer(DFT_signal.get(), static_input_shapes, static_output_shapes, constData);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));

    // DFT StaticShape with one constData
    int32_t ptr1[1] = {2};
    constData[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, ptr1);
    static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    shape_infer(DFT_signal.get(), static_input_shapes, static_output_shapes, constData);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 320, 512, 2}));

    // DFT StaticShape with two constData
    int32_t ptr2[4] = {1, 2, 3, 4};
    constData[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, ptr2);
    constData[2] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, ptr2 + 2);
    static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    shape_infer(DFT_signal.get(), static_input_shapes, static_output_shapes, constData);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 3, 4, 2}));
}

TEST(StaticShapeInferenceTest, IDFTTest) {
    auto input_shape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto axes_shape = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {1, 2});
    auto signal = op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{2}, {512, 100});
    auto IDFT = std::make_shared<ov::op::v7::IDFT>(input_shape, axes_shape);
    auto IDFT_signal = std::make_shared<ov::op::v7::IDFT>(input_shape, axes_shape, signal);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_infer(IDFT.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 320, 320, 2}));

    std::vector<PartialShape> input_shapes = {PartialShape{1, 320, 320, 2}, PartialShape{2}},
                              output_shapes = {PartialShape{}};
    shape_infer(IDFT.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, 320, 320, 2}));

    static_input_shapes = {StaticShape{1, 320, 320, 2}, StaticShape{2}, StaticShape{2}};
    static_output_shapes = {StaticShape{}};
    shape_infer(IDFT_signal.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 512, 100, 2}));

    input_shapes = {PartialShape{1, 320, 320, 2}, PartialShape{2}, PartialShape{2}};
    output_shapes = {PartialShape{}};
    shape_infer(IDFT_signal.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes[0], PartialShape({1, 512, 100, 2}));
}