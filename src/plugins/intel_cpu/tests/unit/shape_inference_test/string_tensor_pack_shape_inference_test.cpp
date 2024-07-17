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
                                                              std::vector<std::string>      // symbols
                                                              >> {};

TEST_P(StringTensorPackStaticTestSuite, StringTensorPackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& indices_shape = std::get<0>(param);
    const auto& begins_param = std::get<1>(param);
    const auto& ends_param = std::get<2>(param);
    const auto& symbols_param = std::get<3>(param);

    const auto begins = std::make_shared<Constant>(element::i64, indices_shape, begins_param);
    const auto ends = std::make_shared<Constant>(element::i64, indices_shape, ends_param);
    const auto symbols = std::make_shared<Constant>(element::string, Shape{symbols_param.size()}, symbols_param);

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
            std::vector<std::string>{u8"\\x49", u8"\\x6e", u8"\\x74", u8"\\x65", u8"\\x6c"}),
        // "Intel", "OpenVINO"
        std::make_tuple(
            Shape{2},
            std::vector<size_t>{0, 5},
            std::vector<size_t>{5, 13},
            std::vector<std::string>{u8"\\x49", u8"\\x6e", u8"\\x74", u8"\\x65", u8"\\x6c", u8"\\x4f",
                                     u8"\\x70", u8"\\x65", u8"\\x6e", u8"\\x56", u8"\\x49", u8"\\x4e", u8"\\x4f"}),
        // " "
        std::make_tuple(
            Shape{1},
            std::vector<size_t>{0},
            std::vector<size_t>{0},
            std::vector<std::string>{u8"\\x20"}),
        // ""
        std::make_tuple(
            Shape{0},
            std::vector<size_t>{},
            std::vector<size_t>{},
            std::vector<std::string>{}),
        // (2, 2) shape; "1", "2", "3", "4"
        std::make_tuple(
            Shape{2, 2},
            std::vector<size_t>{0, 1, 2, 3},
            std::vector<size_t>{1, 2, 3, 4},
            std::vector<std::string>{u8"\\x31", u8"\\x32", u8"\\x33", u8"\\x34"}),
        // (1, 2) shape; "1", "2"
        std::make_tuple(
            Shape{1, 2},
            std::vector<size_t>{0, 1},
            std::vector<size_t>{1, 2},
            std::vector<std::string>{u8"\\x31", u8"\\x32"}),
        // skipped symbols; "1", "9"
        std::make_tuple(
            Shape{2},
            std::vector<size_t>{0, 8},
            std::vector<size_t>{1, 9},
            std::vector<std::string>{u8"\\x31", u8"\\x32", u8"\\x33", u8"\\x34", u8"\\x35", u8"\\x36", u8"\\x37", u8"\\x38", u8"\\x3"}),
        // mixed strings; "1", "", " ", "4"
        std::make_tuple(
            Shape{2, 2},
            std::vector<size_t>{0, 0, 1, 2},
            std::vector<size_t>{1, 1, 2, 3},
            std::vector<std::string>{u8"\\x31", "", u8"\\x20", u8"\\x34"})
    )
);
