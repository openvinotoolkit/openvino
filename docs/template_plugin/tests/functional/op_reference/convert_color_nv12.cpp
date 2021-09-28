// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/function.hpp>
#include <tuple>
#include <openvino/op/nv12_to_rgb.hpp>

#include "base_reference_test.hpp"

using namespace ov;
using namespace InferenceEngine;
using namespace reference_tests;

struct ConvertColorNV12Params {
    ConvertColorNV12Params(std::vector<uint8_t>&& inp,
                           const ov::Shape& inp_shape,
                           std::vector<uint8_t>&& exp, const ov::Shape& exp_shape, std::string&& name)
        : input(std::move(inp)), input_shape(inp_shape),
          expected(std::move(exp)), expected_shape(exp_shape), title(std::move(name)) {}
    std::vector<uint8_t> input;
    ov::Shape input_shape;
    std::vector<uint8_t> expected;
    ov::Shape expected_shape;
    std::string title;
};

using TestNV12Param = std::tuple<ConvertColorNV12Params, bool, bool>;

class ReferenceConvertColorNV12LayerTest : public testing::TestWithParam<TestNV12Param>, public CommonReferenceTest {
public:
    void SetUp() override {
        threshold = 2.f;
    }
    static std::string getTestCaseName(const testing::TestParamInfo<TestNV12Param>& obj) {
        std::ostringstream result;
        result << std::get<0>(obj.param).title;
        result << "_onePlane= " << std::get<1>(obj.param);
        result << "_rgb= " << std::get<2>(obj.param);
        return result.str();
    }

public:
    static std::shared_ptr<Function> CreateFunction(const Tensor& input,
                                                    bool rgb) {
        const auto in = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        std::shared_ptr<Node> conv;
        if (rgb) {
            conv = std::make_shared<op::v8::NV12toRGB>(in);
        } else {
            conv = std::make_shared<op::v8::NV12toBGR>(in);
        }
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Function>(ResultVector{res}, ParameterVector {in});
    }

    static std::shared_ptr<Function> CreateFunction2(const Tensor& input1, const Tensor& input2,
                                                    bool rgb) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input1.type, input1.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(input2.type, input2.shape);
        std::shared_ptr<Node> conv;
        if (rgb) {
            conv = std::make_shared<op::v8::NV12toRGB>(in1, in2);
        } else {
            conv = std::make_shared<op::v8::NV12toBGR>(in1, in2);
        }
        auto res = std::make_shared<op::v0::Result>(conv);
        return std::make_shared<Function>(ResultVector{res}, ParameterVector {in1, in2});
    }
};

TEST_P(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_u8) {
    auto params = std::get<0>(GetParam());
    auto single_plane = std::get<1>(GetParam());
    if (single_plane) {
        Tensor inp_tensor_u8(params.input_shape, element::u8, params.input);
        function = CreateFunction(inp_tensor_u8, std::get<2>(GetParam()));
        inputData = {inp_tensor_u8.data};
    } else {
        auto shape_y =
                Shape({params.input_shape[0], params.input_shape[1] * 2 / 3, params.input_shape[2], 1});
        auto shape_uv = Shape(
                {params.input_shape[0], params.input_shape[1] / 3, params.input_shape[2] / 2, 2});

        std::vector<uint8_t> input1, input2, expected_result;
        std::copy(params.input.begin(), params.input.begin() + shape_size(shape_y), std::back_inserter(input1));
        std::copy(params.input.begin() + shape_size(shape_y), params.input.end(), std::back_inserter(input2));
        std::copy(params.expected.begin(), params.expected.end(), std::back_inserter(expected_result));
        Tensor inp_tensor1_u8(shape_y, element::u8, input1);
        Tensor inp_tensor2_u8(shape_uv, element::u8, input2);
        function = CreateFunction2(inp_tensor1_u8, inp_tensor2_u8, std::get<2>(GetParam()));
        inputData = {inp_tensor1_u8.data, inp_tensor2_u8.data};
    }
    if (std::get<2>(GetParam())) {
        Tensor exp_tensor_u8(params.expected_shape, element::u8, params.expected);
        refOutData = {exp_tensor_u8.data};
    } else {
        std::vector<uint8_t> exp_bgr = params.expected;
        for (size_t i = 0; i < params.expected.size(); i += 3) {
            exp_bgr[i] = params.expected[i+2];
            exp_bgr[i+2] = params.expected[i];
        }
        Tensor exp_tensor_u8(params.expected_shape, element::u8, exp_bgr);
        refOutData = {exp_tensor_u8.data};
    }
    Exec();
}

TEST_P(ReferenceConvertColorNV12LayerTest, CompareWithHardcodedRefs_fp32) {
    auto params = std::get<0>(GetParam());
    auto single_plane = std::get<1>(GetParam());
    std::vector<float> expected_result;
    std::copy(params.expected.begin(), params.expected.end(), std::back_inserter(expected_result));
    if (single_plane) {
        std::vector<float> input;
        std::copy(params.input.begin(), params.input.end(), std::back_inserter(input));
        Tensor inp_tensor_f32(params.input_shape, element::f32, input);
        function = CreateFunction(inp_tensor_f32, std::get<2>(GetParam()));
        inputData = {inp_tensor_f32.data};
    } else {
        auto shape_y =
                Shape({params.input_shape[0], params.input_shape[1] * 2 / 3, params.input_shape[2], 1});
        auto shape_uv = Shape(
                {params.input_shape[0], params.input_shape[1] / 3, params.input_shape[2] / 2, 2});

        std::vector<float> input1, input2;
        std::copy(params.input.begin(), params.input.begin() + shape_size(shape_y), std::back_inserter(input1));
        std::copy(params.input.begin() + shape_size(shape_y), params.input.end(), std::back_inserter(input2));
        Tensor inp_tensor1_f32(shape_y, element::f32, input1);
        Tensor inp_tensor2_f32(shape_uv, element::f32, input2);
        function = CreateFunction2(inp_tensor1_f32, inp_tensor2_f32, std::get<2>(GetParam()));
        inputData = {inp_tensor1_f32.data, inp_tensor2_f32.data};
    }
    if (std::get<2>(GetParam())) {
        Tensor exp_tensor_f32(params.expected_shape, element::f32, expected_result);
        refOutData = {exp_tensor_f32.data};
    } else {
        std::vector<float> exp_bgr = expected_result;
        for (size_t i = 0; i < params.expected.size(); i += 3) {
            exp_bgr[i] = params.expected[i+2];
            exp_bgr[i+2] = params.expected[i];
        }
        Tensor exp_tensor_f32(params.expected_shape, element::f32, exp_bgr);
        refOutData = {exp_tensor_f32.data};
    }
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Convert_color_nv12_With_Hardcoded_Refs, ReferenceConvertColorNV12LayerTest,
    ::testing::Combine(
            ::testing::Values(
            ConvertColorNV12Params(
                    std::vector<uint8_t> {0x51, 0x51, 0x51, 0x51, 0xf0, 0x5a},
                    Shape{1, 3, 2, 1},
                    std::vector<uint8_t> {0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0},
                    Shape{1, 2, 2, 3},
                    "red2x2"),
            ConvertColorNV12Params(
                    std::vector<uint8_t> {0x91, 0x91, 0x91, 0x91, 0x22, 0x36},
                    Shape{1, 3, 2, 1},
                    std::vector<uint8_t> {0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0},
                    Shape{1, 2, 2, 3},
                    "green2x2"),
            ConvertColorNV12Params(
                    std::vector<uint8_t> {0x29, 0x29, 0x29, 0x29, 0x6e, 0xf0},
                    Shape{1, 3, 2, 1},
                    std::vector<uint8_t> {0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff},
                    Shape{1, 2, 2, 3},
                    "blue2x2"),
            ConvertColorNV12Params(
                    std::vector<uint8_t> {0x91, 0x91, 0x51, 0x51, 0x91, 0x91, 0x51, 0x51, 0x29, 0x29, 0x91, 0x5f,
                                          0x29, 0x29, 0x29, 0x29, 0x22, 0x36, 0xf0, 0x5a, 0x6e, 0xf0, 0x76, 0x9f},
                    Shape{1, 6, 4, 1},
                    std::vector<uint8_t> {0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0x00,
                                          0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0xff, 0x00, 0x00,
                                          0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 74,   87,   153,  132,  145,  211,
                                          0x00, 0x00, 0xff, 0x00, 0x00, 0xff, 0x1b, 0x18, 0x5a, 0x1b, 0x18, 0x5a},
                    Shape{1, 4, 4, 3},
                    "color4x4"),
            ConvertColorNV12Params(
                    std::vector<uint8_t> {0x51, 0x51, 0x51, 0x51, 0xf0, 0x5a,
                                          0x91, 0x91, 0x91, 0x91, 0x22, 0x36,
                                          0x29, 0x29, 0x29, 0x29, 0x6e, 0xf0},
                    Shape{3, 3, 2, 1},
                    std::vector<uint8_t> {0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0,
                                          0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0,
                                          0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff},
                    Shape{3, 2, 2, 3},
                    "batch2x2")),
                    ::testing::Bool(),
                    ::testing::Bool()
),
    ReferenceConvertColorNV12LayerTest::getTestCaseName);
