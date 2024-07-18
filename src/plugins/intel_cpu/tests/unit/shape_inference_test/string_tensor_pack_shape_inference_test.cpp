// Copyright (C) 2018-2024 Intel Corporation
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

class StringTensorPackStaticTestSuite : public ::testing::TestWithParam<std::tuple<
                                                              Shape,                        // begins/ends indices shape
                                                              std::vector<size_t>,          // begins
                                                              std::vector<size_t>,          // ends
                                                              std::vector<uint8_t>          // symbols
                                                              >> {};

TEST_P(StringTensorPackStaticTestSuite, StringTensorPackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& indices_shape = std::get<0>(param);
    const auto& begins_param = std::get<1>(param);
    const auto& ends_param = std::get<2>(param);
    const auto& symbols_param = std::get<3>(param);

    const auto begins = std::make_shared<Constant>(element::i64, indices_shape, begins_param);
    const auto ends = std::make_shared<Constant>(element::i64, indices_shape, ends_param);
    const auto symbols = std::make_shared<Constant>(element::u8, Shape{symbols_param.size()}, symbols_param);

    const auto input_shapes = ShapeVector{indices_shape, indices_shape, Shape{symbols_param.size()}};
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
        std::make_tuple(
            Shape{1},
            std::vector<size_t>{0},
            std::vector<size_t>{5},
            std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c}),
        // "Intel", "OpenVINO"
        std::make_tuple(
            Shape{2},
            std::vector<size_t>{0, 5},
            std::vector<size_t>{5, 13},
            std::vector<uint8_t>{0x49, 0x6e, 0x74, 0x65, 0x6c, 0x4f,
                                 0x70, 0x65, 0x6e, 0x56, 0x49, 0x4e, 0x4f}),
        // " "
        std::make_tuple(
            Shape{1},
            std::vector<size_t>{0},
            std::vector<size_t>{0},
            std::vector<uint8_t>{0x20}),
        // ""
        std::make_tuple(
            Shape{0},
            std::vector<size_t>{},
            std::vector<size_t>{},
            std::vector<uint8_t>{}),
        // (2, 2) shape; "1", "2", "3", "4"
        std::make_tuple(
            Shape{2, 2},
            std::vector<size_t>{0, 1, 2, 3},
            std::vector<size_t>{1, 2, 3, 4},
            std::vector<uint8_t>{0x31, 0x32, 0x33, 0x34}),
        // (1, 2) shape; "1", "2"
        std::make_tuple(
            Shape{1, 2},
            std::vector<size_t>{0, 1},
            std::vector<size_t>{1, 2},
            std::vector<uint8_t>{0x31, 0x32}),
        // skipped symbols; "1", "9"
        std::make_tuple(
            Shape{2},
            std::vector<size_t>{0, 8},
            std::vector<size_t>{1, 9},
            std::vector<uint8_t>{0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x3}),
        // mixed strings; "1", "", " ", "4"
        std::make_tuple(
            Shape{2, 2},
            std::vector<size_t>{0, 1, 1, 2},
            std::vector<size_t>{1, 1, 2, 3},
            std::vector<uint8_t>{0x31, 0x20, 0x34})
    )
);
