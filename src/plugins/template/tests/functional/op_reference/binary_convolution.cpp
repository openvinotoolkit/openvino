// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/binary_convolution.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct BinaryConvolutionParams {
    template <class T>
    BinaryConvolutionParams(const Shape& inputShape,
                            const Shape& filterShape,
                            const Shape& outputShape,
                            const element::Type& iType,
                            const std::vector<T>& iValues,
                            const std::vector<uint8_t>& filterValues,
                            const std::vector<T>& oValues,
                            const Strides& strides,
                            const CoordinateDiff& padBegin,
                            const CoordinateDiff& padEnd,
                            const Strides& dialations,
                            const float padValue = 0,
                            const std::string& test_name = "")
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(inputShape, iType, iValues)),
          filterData(filterValues),
          refData(CreateTensor(outputShape, iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations),
          padValue(padValue) {}

    Shape inputShape;
    Shape filterShape;
    Shape outputShape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    std::vector<uint8_t> filterData;
    ov::Tensor refData;
    ov::Strides strides;
    ov::CoordinateDiff padBegin;
    ov::CoordinateDiff padEnd;
    ov::Strides dialations;
    ov::op::v1::BinaryConvolution::BinaryConvolutionMode mode =
        ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    float padValue;
    std::string testcaseName;
};

class ReferenceBinaryConvolutionLayerTest : public testing::TestWithParam<BinaryConvolutionParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params, params.filterData);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<BinaryConvolutionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "filterShape=" << param.filterShape << "_";
        result << "outputShape=" << param.outputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "strides=" << param.strides << "_";
        result << "padBegin=" << param.padBegin << "_";
        result << "padEnd=" << param.padEnd << "_";
        result << "dialations=" << param.dialations << "_";
        if (param.testcaseName != "") {
            result << "padValue=" << param.padValue << "_";
            result << param.testcaseName;
        } else {
            result << "padValue=" << param.padValue;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const BinaryConvolutionParams& params,
                                                 const std::vector<uint8_t>& filterData) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};
        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        auto filter = std::make_shared<op::v0::Constant>(ov::element::u1, params.filterShape, &filterData[0]);
        const auto BinaryConvolution = std::make_shared<op::v1::BinaryConvolution>(in,
                                                                                   filter,
                                                                                   params.strides,
                                                                                   params.padBegin,
                                                                                   params.padEnd,
                                                                                   params.dialations,
                                                                                   params.mode,
                                                                                   params.padValue,
                                                                                   auto_pad);
        return std::make_shared<ov::Model>(NodeVector{BinaryConvolution}, ParameterVector{in});
    }
};

TEST_P(ReferenceBinaryConvolutionLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<BinaryConvolutionParams> generateBinaryConvolutionParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<BinaryConvolutionParams> binaryConvolutionParams{
        // --------------------- 2D BinaryConvolution ------------------------------------------
        BinaryConvolutionParams(Shape{1, 1, 4, 4},
                                Shape{1, 1, 3, 3},
                                Shape{1, 1, 2, 2},
                                IN_ET,
                                std::vector<T>{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1},
                                std::vector<uint8_t>{0xAA, 0x80},  // 10101010 10000000
                                std::vector<T>{1, 1, 3, -1},
                                {1, 1},
                                {0, 0},
                                {0, 0},
                                {1, 1}),
        BinaryConvolutionParams(Shape{1, 1, 4, 4},
                                Shape{1, 1, 3, 3},
                                Shape{1, 1, 4, 4},
                                IN_ET,
                                std::vector<T>{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1},
                                std::vector<uint8_t>{0xAA, 0x80},  // 10101010 10000000
                                std::vector<T>{1, -3, -1, 1, -3, 1, 1, -5, -3, 3, -1, 1, 1, -5, 1, -3},
                                {1, 1},
                                {1, 1},
                                {1, 1},
                                {1, 1}),
        BinaryConvolutionParams(Shape{1, 1, 4, 4},
                                Shape{1, 1, 3, 3},
                                Shape{1, 1, 4, 4},
                                IN_ET,
                                std::vector<T>{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1},
                                std::vector<uint8_t>{0xAA, 0x80},  // 10101010 10000000
                                std::vector<T>{3, -1, 1, 3, -1, 1, 1, -3, -1, 3, -1, 3, 3, -3, 3, -1},
                                {1, 1},
                                {1, 1},
                                {1, 1},
                                {1, 1},
                                1.0f),
        BinaryConvolutionParams(
            Shape{1, 1, 5, 5},
            Shape{1, 1, 3, 3},
            Shape{1, 1, 2, 2},
            IN_ET,
            std::vector<T>{0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1},
            std::vector<uint8_t>{0x2E, 0x00},  // 10101010 10000000
            std::vector<T>{-1, 3, 1, 1},
            {2, 2},
            {0, 0},
            {0, 0},
            {1, 1}),
        BinaryConvolutionParams(
            Shape{1, 1, 7, 7},
            Shape{1, 1, 3, 3},
            Shape{1, 1, 3, 3},
            IN_ET,
            std::vector<T>{1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                           1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0},
            std::vector<uint8_t>{0x6B, 0x00},  // 10101010 10000000
            std::vector<T>{-5, -3, -5, 5, 1, 3, -1, -1, 3},
            {1, 1},
            {0, 0},
            {0, 0},
            {2, 2}),
        BinaryConvolutionParams(
            Shape{1, 1, 7, 7},
            Shape{1, 1, 3, 3},
            Shape{1, 1, 4, 4},
            IN_ET,
            std::vector<T>{1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                           1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0},
            std::vector<uint8_t>{0x6B, 0x00},  // 10101010 10000000
            std::vector<T>{1, 1, -1, 1, 1, -5, -5, 5, 3, -1, 3, 3, -1, -1, 3, -3},
            {2, 2},
            {2, 2},
            {2, 2},
            {2, 2}),
        BinaryConvolutionParams(
            Shape{1, 1, 7, 7},
            Shape{1, 1, 3, 3},
            Shape{1, 1, 4, 4},
            IN_ET,
            std::vector<T>{1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                           1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0},
            std::vector<uint8_t>{0x6B, 0x00},  // 10101010 10000000
            std::vector<T>{3, 3, 1, -1, -1, -5, -5, 3, 1, -1, 3, 1, -3, 1, 5, -1},
            {2, 2},
            {2, 2},
            {2, 2},
            {2, 2},
            1.0f),
        BinaryConvolutionParams(Shape{1, 2, 4, 4},
                                Shape{1, 2, 3, 3},
                                Shape{1, 1, 2, 2},
                                IN_ET,
                                std::vector<T>{// channel 1
                                               1,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               1,
                                               1,
                                               0,
                                               1,
                                               1,
                                               // channel 2
                                               0,
                                               1,
                                               1,
                                               0,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               0,
                                               0,
                                               1,
                                               0,
                                               0},
                                std::vector<uint8_t>{0xAA, 0xAA, 0x80},  // 10101010 10000000
                                std::vector<T>{2, 2, 6, -2},
                                {1, 1},
                                {0, 0},
                                {0, 0},
                                {1, 1}),
        BinaryConvolutionParams(Shape{2, 1, 4, 4},
                                Shape{1, 1, 3, 3},
                                Shape{2, 1, 2, 2},
                                IN_ET,
                                std::vector<T>{// batch 1
                                               1,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               0,
                                               1,
                                               1,
                                               0,
                                               1,
                                               1,
                                               // batch 2
                                               0,
                                               0,
                                               0,
                                               0,
                                               1,
                                               1,
                                               1,
                                               0,
                                               1,
                                               1,
                                               0,
                                               1,
                                               1,
                                               0,
                                               1,
                                               0},
                                std::vector<uint8_t>{0xAA, 0x80},  // 10101010 10000000
                                std::vector<T>{                    // batch 1
                                               1,
                                               1,
                                               3,
                                               -1,
                                               // batch 2
                                               -3,
                                               3,
                                               5,
                                               -7},
                                {1, 1},
                                {0, 0},
                                {0, 0},
                                {1, 1})};
    return binaryConvolutionParams;
}

std::vector<BinaryConvolutionParams> generateBinaryConvolutionCombinedParams() {
    const std::vector<std::vector<BinaryConvolutionParams>> binaryConvolutionTypeParams{
        generateBinaryConvolutionParams<element::Type_t::f32>(),
        generateBinaryConvolutionParams<element::Type_t::f16>(),
        generateBinaryConvolutionParams<element::Type_t::i64>(),
        generateBinaryConvolutionParams<element::Type_t::i32>()};
    std::vector<BinaryConvolutionParams> combinedParams;

    for (const auto& params : binaryConvolutionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BinaryConvolution_With_Hardcoded_Refs,
                         ReferenceBinaryConvolutionLayerTest,
                         testing::ValuesIn(generateBinaryConvolutionCombinedParams()),
                         ReferenceBinaryConvolutionLayerTest::getTestCaseName);

}  // namespace
