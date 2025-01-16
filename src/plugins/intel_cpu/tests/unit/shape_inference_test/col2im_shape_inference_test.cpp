// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

class Col2ImStaticTestSuite : public ::testing::TestWithParam<std::tuple<ov::Shape,      // data shape
                                                              std::vector<int32_t>,     // output_size values
                                                              std::vector<int32_t>,     // kernel_size values
                                                              Strides,                  // strides
                                                              Strides,                  // dilations
                                                              ov::Shape,                // pads_begin
                                                              ov::Shape,                // pads_end
                                                              ov::Shape>> {};           // expected output shape

class Col2ImStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v15::Col2Im> {};

TEST_F(Col2ImStaticShapeInferenceTest, kernel_size_and_output_size_from_tensor_accessor) {
    const auto data = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto output_size = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto kernel_size = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto strides = Strides{2, 2};
    const auto dilations = Strides{2, 2};
    const auto pads_begin = Shape{2, 2};
    const auto pads_end = Shape{2, 2};
    const auto op = make_op(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);

    int64_t output_size_val[] = {32, 32};
    int64_t kernel_size_val[] = {2, 2};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{1, {element::i64, Shape{2}, output_size_val}},
                                                           {2, {element::i64, Shape{2}, kernel_size_val}}};

    const auto input_shapes = ShapeVector{Shape{3, 12, 289}, Shape{2}, Shape{2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({3, 3, 32, 32}));
}

TEST_P(Col2ImStaticTestSuite, Col2ImStaticShapeInference) {
    const auto& param = GetParam();
    const auto& data_shape = std::get<0>(param);
    const auto& output_size_val = std::get<1>(param);
    const auto& kernel_size_val = std::get<2>(param);
    const auto& strides = std::get<3>(param);
    const auto& dilations = std::get<4>(param);
    const auto& pads_begin = std::get<5>(param);
    const auto& pads_end = std::get<6>(param);
    const auto& expected_output_shape = std::get<7>(param);

    const auto data = std::make_shared<Parameter>(element::i64, data_shape);
    const auto output_size = std::make_shared<Constant>(element::i64, Shape{2}, output_size_val);
    const auto kernel_size = std::make_shared<Constant>(element::i64, Shape{2}, kernel_size_val);
    const auto op = std::make_shared<op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);
    const auto input_shapes = ShapeVector{data_shape, Shape{2}, Shape{2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape(expected_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    Col2ImStaticShapeInferenceTests,
    Col2ImStaticTestSuite,
    ::testing::Values(
        std::make_tuple(
            Shape{3, 12, 81},               // data shape
            std::vector<int32_t>{16, 16},   // output_size values
            std::vector<int32_t>{2, 2},     // kernel_size values
            Strides{2, 2},                  // strides
            Strides{2, 2},                  // dilations
            Shape{2, 2},                    // pads_begin
            Shape{2, 2},                    // pads_end
            Shape{3, 3, 16, 16}),           // expected output shape
        std::make_tuple(
            Shape{12, 81},                  // data shape
            std::vector<int32_t>{16, 16},   // output_size values
            std::vector<int32_t>{2, 2},     // kernel_size values
            Strides{2, 2},                  // strides
            Strides{2, 2},                  // dilations
            Shape{2, 2},                    // pads_begin
            Shape{2, 2},                    // pads_end
            Shape{3, 16, 16}),              // expected output shape
        std::make_tuple(
            Shape{3, 12, 225},              // data shape
            std::vector<int32_t>{16, 16},   // output_size values
            std::vector<int32_t>{2, 2},     // kernel_size values
            Strides{1, 1},                  // strides
            Strides{1, 1},                  // dilations
            Shape{0, 0},                    // pads_begin
            Shape{0, 0},                    // pads_end
            Shape{3, 3, 16, 16}),           // expected output shape
        std::make_tuple(
            Shape{1, 27, 49},               // data shape
            std::vector<int32_t>{16, 16},   // output_size values
            std::vector<int32_t>{3, 3},     // kernel_size values
            Strides{2, 2},                  // strides
            Strides{2, 2},                  // dilations
            Shape{1, 1},                    // pads_begin
            Shape{1, 1},                    // pads_end
            Shape{1, 3, 16, 16}),           // expected output shape
        std::make_tuple(
            Shape{1, 18, 104},              // data shape
            std::vector<int32_t>{16, 16},   // output_size values
            std::vector<int32_t>{2, 3},     // kernel_size values
            Strides{2, 1},                  // strides
            Strides{2, 2},                  // dilations
            Shape{1, 0},                    // pads_begin
            Shape{0, 1},                    // pads_end
            Shape{1, 3, 16, 16}),           // expected output shape
        std::make_tuple(
            Shape{12, 12, 324},             // data shape
            std::vector<int32_t>{32, 32},   // output_size values
            std::vector<int32_t>{2, 2},     // kernel_size values
            Strides{2, 2},                  // strides
            Strides{2, 2},                  // dilations
            Shape{3, 3},                    // pads_begin
            Shape{3, 3},                    // pads_end
            Shape{12, 3, 32, 32})           // expected output shape
    )
);
