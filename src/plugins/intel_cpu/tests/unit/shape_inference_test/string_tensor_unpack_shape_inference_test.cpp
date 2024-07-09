// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"
#include "string_tensor_unpack_shape_inference.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;

static size_t get_character_count(const std::vector<std::string>& vec) {
    size_t count = 0;
    for (const auto& str : vec) {
        count += str.size();
    }
    return count;
}

class StringTensorUnpackStaticTestSuite : public ::testing::TestWithParam<std::tuple<
                                                              std::vector<std::string>,         // input data
                                                              Shape                             // input shape
                                                              >> {};

class StringTensorUnpackStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v15::StringTensorUnpack> {};

TEST_P(StringTensorUnpackStaticTestSuite, StringTensorUnpackStaticShapeInference) {
    const auto& param = GetParam();
    const auto& input_string = std::get<0>(param);
    const auto& input_shape = std::get<1>(param);

    const auto data = std::make_shared<Constant>(element::string, input_shape, input_string);
    const auto op = std::make_shared<op::v15::StringTensorUnpack>(data);
    const auto input_shapes = ShapeVector{input_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 3);
    EXPECT_EQ(output_shapes.front(), StaticShape(input_shape));
    EXPECT_EQ(output_shapes[1], StaticShape(input_shape));
}

INSTANTIATE_TEST_SUITE_P(
    StringTensorUnpackStaticShapeInferenceTests,
    StringTensorUnpackStaticTestSuite,
    ::testing::Values(
        std::make_tuple(
            std::vector<std::string>{"Intel", "OpenVINO"},
            Shape{2})
    )
);
