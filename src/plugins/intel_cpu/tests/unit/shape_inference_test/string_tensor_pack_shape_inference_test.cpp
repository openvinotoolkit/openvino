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
                                                              Shape,                        // input shape
                                                              std::vector<size_t>,          // begins
                                                              std::vector<size_t>,          // ends
                                                              std::string                   // symbols
                                                              >> {};

TEST_P(StringTensorPackStaticTestSuite, StringTensorPackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& input_shape = std::get<0>(param);
    const auto& begins_param = std::get<1>(param);
    const auto& ends_param = std::get<2>(param);
    const auto& symbols_param = std::get<3>(param);

    const auto begins = std::make_shared<Constant>(element::i64, input_shape, begins_param);
    const auto ends = std::make_shared<Constant>(element::i64, input_shape, ends_param);
    const auto symbols = std::make_shared<Constant>(element::u8, Shape{1}, std::vector<std::string>{symbols_param});
    const auto op = std::make_shared<op::v15::StringTensorPack>(begins, ends, symbols);
    const auto input_shapes = ShapeVector{input_shape, input_shape, Shape{1}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0], StaticShape(input_shape));
}

INSTANTIATE_TEST_SUITE_P(
    StringTensorPackStaticShapeInferenceTests,
    StringTensorPackStaticTestSuite,
    ::testing::Values(
        std::make_tuple(
            Shape{1},
            std::vector<size_t>{0},
            std::vector<size_t>{5},
            u8"\\x49\\x6e\\x74\\x65\\x6c")
    )
);
