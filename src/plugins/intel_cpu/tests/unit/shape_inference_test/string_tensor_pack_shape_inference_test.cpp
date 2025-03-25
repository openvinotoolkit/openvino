// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "string_tensor_pack_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;

class StringTensorPackStaticTestSuite
    : public ::testing::TestWithParam<std::tuple<ov::Shape,             // begins/ends indices shape
                                                 std::vector<int32_t>,  // begins
                                                 std::vector<int32_t>,  // ends
                                                 std::vector<uint8_t>   // symbols
                                                 >> {};

TEST_P(StringTensorPackStaticTestSuite, StringTensorPackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& indices_shape = std::get<0>(param);
    const auto& begins_param = std::get<1>(param);
    const auto& ends_param = std::get<2>(param);
    const auto& symbols_param = std::get<3>(param);

    const auto begins = std::make_shared<Constant>(element::i32, indices_shape, begins_param);
    const auto ends = std::make_shared<Constant>(element::i32, indices_shape, ends_param);
    const auto symbols = std::make_shared<Constant>(element::u8, ov::Shape{symbols_param.size()}, symbols_param);

    const auto input_shapes = StaticShapeVector{indices_shape, indices_shape, ov::Shape{symbols_param.size()}};
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto op = std::make_shared<op::v15::StringTensorPack>(begins, ends, symbols);
    auto shape_infer = make_shape_inference(op);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape(indices_shape));
}

INSTANTIATE_TEST_SUITE_P(
    StringTensorPackStaticShapeInferenceTests,
    StringTensorPackStaticTestSuite,
    ::testing::Values(
        // "Intel"
        std::make_tuple(ov::Shape{1},
                        std::vector<int32_t>{0},
                        std::vector<int32_t>{5},
                        std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c}),
        // "Intel", "OpenVINO"
        std::make_tuple(
            ov::Shape{2},
            std::vector<int32_t>{0, 5},
            std::vector<int32_t>{5, 13},
            std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f, 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f}),
        // " "
        std::make_tuple(ov::Shape{1}, std::vector<int32_t>{0}, std::vector<int32_t>{0}, std::vector<uint8_t>{0x20}),
        // ""
        std::make_tuple(ov::Shape{0}, std::vector<int32_t>{}, std::vector<int32_t>{}, std::vector<uint8_t>{}),
        // (2, 2) shape; "1", "2", "3", "4"
        std::make_tuple(ov::Shape{2, 2},
                        std::vector<int32_t>{0, 1, 2, 3},
                        std::vector<int32_t>{1, 2, 3, 4},
                        std::vector<uint8_t>{0x31, 0x32, 0x33, 0x34}),
        // (1, 2) shape; "1", "2"
        std::make_tuple(ov::Shape{1, 2},
                        std::vector<int32_t>{0, 1},
                        std::vector<int32_t>{1, 2},
                        std::vector<uint8_t>{0x31, 0x32}),
        // skipped symbols; "1", "9"
        std::make_tuple(ov::Shape{2},
                        std::vector<int32_t>{0, 8},
                        std::vector<int32_t>{1, 9},
                        std::vector<uint8_t>{0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x3}),
        // mixed strings; "1", "", " ", "4"
        std::make_tuple(ov::Shape{2, 2},
                        std::vector<int32_t>{0, 1, 1, 2},
                        std::vector<int32_t>{1, 1, 2, 3},
                        std::vector<uint8_t>{0x31, 0x20, 0x34})));

class StringTensorPackStaticShapeInferenceWithTensorAccessorTest: public OpStaticShapeInferenceTest<op::v15::StringTensorPack> {};

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, data_from_tensor_accessor_1) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    int32_t begins[] = {0};
    int32_t ends[] = {5};
    uint8_t symbols[] = {0x49, 0x6e, 0x74, 0x65, 0x6c};
    const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{1}, begins}},
                                                                 {1, {element::i32, ov::Shape{1}, ends}},
                                                                 {2, {element::u8, ov::Shape{5}, symbols}}};

    const auto input_shapes = StaticShapeVector{{1}, {1}, {5}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1}));
}

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, data_from_tensor_accessor_2) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    int32_t begins[] = {0, 1, 2, 3};
    int32_t ends[] = {1, 2, 3, 4};
    uint8_t symbols[] = {0x31, 0x32, 0x33, 0x34};
    const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                 {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                 {2, {element::u8, ov::Shape{4}, symbols}}};

    const auto input_shapes = StaticShapeVector{{2, 2}, {2, 2}, {4}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 2}));
}

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, data_from_tensor_accessor_3) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    int32_t begins[] = {0, 1};
    int32_t ends[] = {1, 2};
    uint8_t symbols[] = {0x31, 0x32};
    const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{1, 2}, begins}},
                                                                 {1, {element::i32, ov::Shape{1, 2}, ends}},
                                                                 {2, {element::u8, ov::Shape{2}, symbols}}};

    const auto input_shapes = StaticShapeVector{{1, 2}, {1, 2}, {2}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({1, 2}));
}

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, data_from_tensor_accessor_4) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    int32_t begins[] = {0, 8};
    int32_t ends[] = {1, 9};
    uint8_t symbols[] = {0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x3};
    const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2}, begins}},
                                                                 {1, {element::i32, ov::Shape{2}, ends}},
                                                                 {2, {element::u8, ov::Shape{9}, symbols}}};

    const auto input_shapes = StaticShapeVector{{2}, {2}, {9}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2}));
}

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, data_from_tensor_accessor_5) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    int32_t begins[] = {0, 1, 1, 2};
    int32_t ends[] = {1, 1, 2, 3};
    uint8_t symbols[] = {0x31, 0x20, 0x34};
    const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                 {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                 {2, {element::u8, ov::Shape{3}, symbols}}};

    const auto input_shapes = StaticShapeVector{{2, 2}, {2, 2}, {3}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape({2, 2}));
}

TEST_F(StringTensorPackStaticShapeInferenceWithTensorAccessorTest, indices_validation) {
    const auto begins_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto ends_param = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());
    const auto symbols_param = std::make_shared<Parameter>(element::u8, ov::PartialShape::dynamic());
    const auto op = make_op(begins_param, ends_param, symbols_param);
    uint8_t symbols[] = {0x31, 0x20, 0x34};
    const auto input_shapes = StaticShapeVector{{2, 2}, {2, 2}, {3}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    {  // negative begins indices
        int32_t begins[] = {-1, 1, 1, 2};
        int32_t ends[] = {1, 1, 2, 3};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("Indices cannot be negative"));
    }
    {  // negative ends indices
        int32_t begins[] = {1, 1, 1, 2};
        int32_t ends[] = {-1, 1, 2, 3};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("Indices cannot be negative"));
    }
    {  // begins out of bounds
        int32_t begins[] = {1, 1, 1, 4};
        int32_t ends[] = {1, 1, 2, 3};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("The biggest index cannot be higher than the amount or characters in symbols input"));
    }
    {  // ends out of bounds
        int32_t begins[] = {1, 1, 1, 3};
        int32_t ends[] = {1, 1, 2, 4};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("The biggest index cannot be higher than the amount or characters in symbols input"));
    }
    {  // unsorted begins
        int32_t begins[] = {1, 3, 1, 2};
        int32_t ends[] = {1, 1, 2, 3};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("Indices must be in ascending order"));
    }
    {  // unsorted ends
        int32_t begins[] = {1, 1, 1, 2};
        int32_t ends[] = {1, 1, 5, 3};
        const auto const_inputs = std::unordered_map<size_t, Tensor>{{0, {element::i32, ov::Shape{2, 2}, begins}},
                                                                     {1, {element::i32, ov::Shape{2, 2}, ends}},
                                                                     {2, {element::u8, ov::Shape{3}, symbols}}};
        OV_EXPECT_THROW(std::ignore = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs)),
                        NodeValidationFailure,
                        testing::HasSubstr("Indices must be in ascending order"));
    }
}
