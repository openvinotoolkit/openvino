// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConvolutionParams {
    template <class IT>
    ConvolutionParams(const Shape& inputShape,
                      const Shape& filterShape,
                      const Shape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues,
                      const std::vector<IT>& filterValues,
                      const std::vector<IT>& oValues,
                      const Strides& strides,
                      const CoordinateDiff& padBegin,
                      const CoordinateDiff& padEnd,
                      const Strides& dialations)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(inputShape, iType, iValues)),
          filterData(CreateTensor(filterShape, iType, filterValues)),
          refData(CreateTensor(outputShape, iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations) {}

    template <class IT>
    ConvolutionParams(const Shape& inputShape,
                      const Shape& filterShape,
                      const Shape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues,
                      const std::vector<IT>& filterValues,
                      const std::vector<IT>& oValues,
                      const Strides& strides,
                      const CoordinateDiff& padBegin,
                      const CoordinateDiff& padEnd,
                      const Strides& dialations,
                      const bool convolutionOutlining)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(inputShape, iType, iValues)),
          filterData(CreateTensor(filterShape, iType, filterValues)),
          refData(CreateTensor(outputShape, iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations),
          convolutionOutlining(convolutionOutlining) {}

    Shape inputShape;
    Shape filterShape;
    Shape outputShape;
    ov::element::Type inType;
    ov::element::Type filterType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor filterData;
    ov::Tensor refData;
    ov::Strides strides;
    ov::CoordinateDiff padBegin;
    ov::CoordinateDiff padEnd;
    ov::Strides dialations;
    bool convolutionOutlining = false;
};

class ReferenceConvolutionLayerTest : public testing::TestWithParam<ConvolutionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionParams>& obj) {
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
        result << "dialations=" << param.dialations;
        if (param.convolutionOutlining == true)
            result << "_convOutlining=" << param.convolutionOutlining;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConvolutionParams& params) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.inType, params.filterShape);
        const auto Convolution = std::make_shared<op::v1::Convolution>(in,
                                                                       filter,
                                                                       params.strides,
                                                                       params.padBegin,
                                                                       params.padEnd,
                                                                       params.dialations,
                                                                       auto_pad);

        if (params.convolutionOutlining == true) {
            const auto Convolution2 = std::make_shared<op::v1::Convolution>(Convolution,
                                                                            filter,
                                                                            params.strides,
                                                                            params.padBegin,
                                                                            params.padEnd,
                                                                            params.dialations,
                                                                            auto_pad);
            return std::make_shared<ov::Model>(NodeVector{Convolution2}, ParameterVector{in, filter});
        } else {
            return std::make_shared<ov::Model>(NodeVector{Convolution}, ParameterVector{in, filter});
        }
    }
};

TEST_P(ReferenceConvolutionLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ConvolutionParams> generateConvolutionI8Params() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ConvolutionParams> convolutionParams{
        // --------------------- 1D convolution ------------------------------------------
        // clang-format off
        ConvolutionParams(Shape{1, 1, 6},
                          Shape{1, 1, 3},
                          Shape{1, 1, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0, 1, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{5, 6, 7, 2},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(Shape{1, 1, 4, 4},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7,
                                         7, 5, 3, 1,
                                         2, 4, 6, 8,
                                         8, 6, 4, 2},
                          std::vector<T>{1, 2, 3,
                                         0, 1, 0,
                                         3, 2, 1},
                          std::vector<T>{47, 69,
                                         70, 48},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 4, 4, 4},
                          Shape{1, 1, 3, 3, 3},
                          Shape{1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1})
    };
    return convolutionParams;
}

template <element::Type_t IN_ET>
std::vector<ConvolutionParams> generateConvolutionFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ConvolutionParams> convolutionParams {
// --------------------- 1D convolution ------------------------------------------
        ConvolutionParams(Shape{1, 1, 6},
                          Shape{1, 1, 3},
                          Shape{1, 1, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0, 1, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{5, 6, 7, 2},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(Shape{1, 1, 4},
                          Shape{1, 1, 3},
                          Shape{1, 1, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{3, 5, 6, 6},
                          {1},
                          {1},
                          {1},
                          {1}),
        ConvolutionParams(Shape{1, 1, 5},
                          Shape{1, 1, 3},
                          Shape{1, 1, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0, 1},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{5, 7},
                          {2},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(Shape{1, 1, 7},
                          Shape{1, 1, 3},
                          Shape{1, 1, 3},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0, 1, 2, 3},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{3, 8, 9},
                          {1},
                          {0},
                          {0},
                          {2}),
        ConvolutionParams(Shape{1, 1, 7},
                          Shape{1, 1, 3},
                          Shape{1, 1, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 3, 0, 1, 2, 3},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{3, 3, 9, 2},
                          {2},
                          {2},
                          {2},
                          {2}),
        ConvolutionParams(Shape{1, 2, 4},
                          Shape{1, 2, 3},
                          Shape{1, 1, 2},
                          IN_ET,
                          std::vector<T>{
                                        // channel 1
                                        1, 3, 2, 1,
                                        // channel 2
                                        2, 2, 3, 1},
                          std::vector<T>{
                                        // channel 1
                                        2, 0, 1,
                                        // channel 2
                                        1, 0, 2},
                          std::vector<T>{12, 11},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(Shape{1, 1, 4},
                          Shape{2, 1, 3},
                          Shape{1, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 2, 1},
                          std::vector<T>{
                                        // filter 1
                                        2, 0, 1,
                                        // filter 2
                                        1, 0, 2},
                          std::vector<T>{
                                        // channel 1
                                        4, 7,
                                        // channel 2
                                        5, 5},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(Shape{2, 1, 4},
                          Shape{1, 1, 3},
                          Shape{2, 1, 2},
                          IN_ET,
                          std::vector<T>{
                                        // batch 1
                                        1, 3, 2, 1,
                                        // batch 2
                                        2, 2, 3, 1},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{
                                        // batch 1
                                        4, 7,
                                        // batch 2
                                        7, 5},
                          {1},
                          {0},
                          {0},
                          {1}),
// --------------------- 2D convolution ------------------------------------------
        ConvolutionParams(Shape{1, 1, 4, 4},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7,
                                         7, 5, 3, 1,
                                         2, 4, 6, 8,
                                         8, 6, 4, 2},
                          std::vector<T>{1, 2, 3,
                                         0, 1, 0,
                                         3, 2, 1},
                          std::vector<T>{47, 69,
                                         70, 48},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 4, 4},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7,
                                         7, 5, 3, 1,
                                         2, 4, 6, 8,
                                         8, 6, 4, 2},
                          std::vector<T>{1, 2, 3,
                                         0, 1, 0,
                                         2, 1, 2},
                          std::vector<T>{18, 28, 20, 14,
                                         28, 47, 67, 40,
                                         51, 60, 40, 23,
                                         24, 34, 44, 24},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 5, 5},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7, 9,
                                         7, 5, 3, 1, 0,
                                         2, 4, 6, 8, 10,
                                         8, 6, 4, 2, 0,
                                         2, 4, 6, 8, 10},
                          std::vector<T>{1, 2, 3,
                                         1, 1, 1,
                                         3, 2, 1},
                          std::vector<T>{57, 94,
                                         66, 102},
                          {2, 2},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 7, 7},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7, 9, 11, 13,
                                         7, 5, 3, 1, -1, -3, -5,
                                         2, 4, 6, 8, 10, 12, 14,
                                         8, 6, 4, 2, 0, -2, -4,
                                         2, 4, 6, 8, 10, 12, 14,
                                         7, 5, 3, 1, -1, -3, -5,
                                         8, 6, 4, 2, 0, -2, -4},
                          std::vector<T>{1, 2, 3,
                                         1, 1, 0,
                                         3, 1, 2},
                          std::vector<T>{78, 106, 134,
                                         44, 16, -12,
                                         80, 84, 88},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {2, 2}),
        ConvolutionParams(Shape{1, 1, 7, 7},
                          Shape{1, 1, 3, 3},
                          Shape{1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7, 9, 11, 13,
                                         7, 5, 3, 1, -1, -3, -5,
                                         2, 4, 6, 8, 10, 12, 14,
                                         8, 6, 4, 2, 0, -2, -4,
                                         2, 4, 6, 8, 10, 12, 14,
                                         7, 5, 3, 1, -1, -3, -5,
                                         8, 6, 4, 2, 0, -2, -4},
                          std::vector<T>{1, 2, 3,
                                         1, 1, 0,
                                         3, 1, 2},
                          std::vector<T>{15, 38, 70, 66,
                                         33, 78, 134, 103,
                                         40, 80, 88, 58,
                                         30, 56, 72, 34},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {2, 2}),
        ConvolutionParams(Shape{1, 2, 4, 4},
                          Shape{1, 2, 3, 3},
                          Shape{1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                         // channel 1
                                         1, 3, 5, 7,
                                         7, 5, 3, 1,
                                         2, 4, 6, 8,
                                         8, 6, 4, 2,
                                         // channel 2
                                         -1, 3, -5, 7,
                                         7, -5, 3, -1,
                                         -2, 4, -6, 8,
                                         8, -6, 4, -2},
                          std::vector<T>{
                                         // channel 1
                                         5, 3, 5,
                                         1, 3, 1,
                                         4, 2, 4,
                                         // channel 2
                                         -5, 3, 5,
                                         1, -3, 1,
                                         4, 2, -4},
                          std::vector<T>{142, 102,
                                         94, 160},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 4, 4},
                          Shape{2, 1, 3, 3},
                          Shape{1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 3, 5, 7,
                                         7, 5, 3, 1,
                                         2, 4, 6, 8,
                                         8, 6, 4, 2},
                          std::vector<T>{
                                         // channel 1
                                         5, 3, 5,
                                         1, 3, 1,
                                         4, 2, 4,
                                         // channel 2
                                         -5, 3, 5,
                                         1, -3, 1,
                                         4, 2, -4},
                          std::vector<T>{
                                         // channel 1
                                         104, 140,
                                         145, 109,
                                         // channel 2
                                         16, 28,
                                         19, 7},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{2, 1, 4, 4},
                          Shape{1, 1, 3, 3},
                          Shape{2, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                         // batch 1
                                         1, 3, 2, 1,
                                         1, 3, 3, 1,
                                         2, 1, 1, 3,
                                         3, 2, 3, 3,
                                         // batch 2
                                         -1, 3, 2, -1,
                                         1, 3, -3, 1,
                                         -2, -1, 1, 3,
                                         3, 2, 3, -3},
                          std::vector<T>{-5, 3, 5,
                                         1, -3, 1,
                                         4, 2, -4},
                          std::vector<T>{
                                         // batch 1
                                         15, -15,
                                         23, 2,
                                         // batch 2
                                         -1, -15,
                                         -5, 6},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
// --------------------- 3D convolution ------------------------------------------
        ConvolutionParams(Shape{1, 1, 4, 4, 4},
                          Shape{1, 1, 3, 3, 3},
                          Shape{1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(Shape{1, 1, 4, 4, 4},
                          Shape{1, 1, 3, 3, 3},
                          Shape{1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     16, 28, 26, 16,
                                     32, 46, 44, 20,
                                     40, 62, 52, 34,
                                     20, 18, 30, 20,
                                     // depth: 2
                                     24, 42, 39, 24,
                                     48, 69, 66, 30,
                                     60, 93, 78, 51,
                                     30, 27, 45, 30,
                                     // depth: 3
                                     24, 42, 39, 24,
                                     48, 69, 66, 30,
                                     60, 93, 78, 51,
                                     30, 27, 45, 30,
                                     // depth: 4
                                     16, 28, 26, 16,
                                     32, 46, 44, 20,
                                     40, 62, 52, 34,
                                     20, 18, 30, 20},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1}),
        ConvolutionParams(Shape{1, 1, 5, 5, 5},
                          Shape{1, 1, 3, 3, 3},
                          Shape{1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1, 2,
                                    1, 3, 3, 1, 2,
                                    2, 1, 1, 3, 2,
                                    3, 2, 3, 3, 2,
                                    3, 2, 3, 3, 2,
                                    // depth: 2
                                    1, 3, 2, 1, 2,
                                    1, 3, 3, 1, 2,
                                    2, 1, 1, 3, 2,
                                    3, 2, 3, 3, 2,
                                    3, 2, 3, 3, 2,
                                    // depth: 3
                                    1, 3, 2, 1, 2,
                                    1, 3, 3, 1, 2,
                                    2, 1, 1, 3, 2,
                                    3, 2, 3, 3, 2,
                                    3, 2, 3, 3, 2,
                                    // depth: 4
                                    1, 3, 2, 1, 2,
                                    1, 3, 3, 1, 2,
                                    2, 1, 1, 3, 2,
                                    3, 2, 3, 3, 2,
                                    3, 2, 3, 3, 2,
                                    // depth: 5
                                    1, 3, 2, 1, 2,
                                    1, 3, 3, 1, 2,
                                    2, 1, 1, 3, 2,
                                    3, 2, 3, 3, 2,
                                    3, 2, 3, 3, 2},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     69, 60,
                                     69, 87,
                                     // depth: 2
                                     69, 60,
                                     69, 87},
                          {2, 2, 2},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(Shape{1, 1, 7, 7, 7},
                          Shape{1, 1, 3, 3, 3},
                          Shape{1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 2
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 3
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 4
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 5
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 6
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    // depth: 7
                                    1, 3, 2, 1, 1, 2, 3,
                                    1, 3, 3, 1, 1, 2, 3,
                                    2, 1, 1, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3,
                                    3, 2, 3, 3, 1, 2, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     10, 18, 20, 16,
                                     38, 40, 54, 30,
                                     38, 42, 52, 30,
                                     36, 30, 30, 20,
                                     // depth: 2
                                     15, 27, 30, 24,
                                     57, 60, 81, 45,
                                     57, 63, 78, 45,
                                     54, 45, 45, 30,
                                     // depth: 3
                                     15, 27, 30, 24,
                                     57, 60, 81, 45,
                                     57, 63, 78, 45,
                                     54, 45, 45, 30,
                                     // depth: 4
                                     10, 18, 20, 16,
                                     38, 40, 54, 30,
                                     38, 42, 52, 30,
                                     36, 30, 30, 20},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2}),
        ConvolutionParams(Shape{1, 2, 4, 4, 4},
                          Shape{1, 2, 3, 3, 3},
                          Shape{1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // depth: 1
                                     138, 132,
                                     186, 156,
                                     // depth: 2
                                     138, 132,
                                     186, 156},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(Shape{1, 1, 4, 4, 4},
                          Shape{2, 1, 3, 3, 3},
                          Shape{1, 2, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // -- filter 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- filter 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // -- out 1 --
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78,
                                     // -- out 2 --
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(Shape{2, 1, 4, 4, 4},
                          Shape{1, 1, 3, 3, 3},
                          Shape{2, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // -- batch 1 --
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // -- batch 2 --
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                     // -- batch 1 --
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78,
                                     // -- batch 2 --
                                     // depth: 1
                                     69, 66,
                                     93, 78,
                                     // depth: 2
                                     69, 66,
                                     93, 78},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
// ----------------------  other tests ------------------------------------------
        ConvolutionParams(Shape{1, 2, 2, 2},
                          Shape{2, 2, 1, 1},
                          Shape{1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 1, 1, 1, 1, 1, 1, 1},
                          std::vector<T>{1, 1, 1, 1},
                          std::vector<T>{4, 4, 4, 4, 4, 4, 4, 4},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          true),
        ConvolutionParams(Shape{1, 2, 2, 2},
                          Shape{2, 2, 1, 1},
                          Shape{1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                          std::vector<T>{3, 3, 3, 3},
                          std::vector<T>{18, 24, 30, 36, 18, 24, 30, 36},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(Shape{1, 1, 2, 2},
                          Shape{1, 1, 1, 1},
                          Shape{1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{1, 2, 3, 4},
                          std::vector<T>{2},
                          std::vector<T>{
                                  0, 0, 0, 0, 0,
                                  0, 2, 4, 0, 0,
                                  0, 6, 8, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0},
                          {1, 1},
                          {1, 1},
                          {2, 2},
                          {1, 1})
    };
    return convolutionParams;
}
// clang-format on

std::vector<ConvolutionParams> generateConvolutionCombinedParams() {
    const std::vector<std::vector<ConvolutionParams>> convolutionTypeParams{
        generateConvolutionFloatParams<element::Type_t::f32>(),
        generateConvolutionFloatParams<element::Type_t::f16>(),
        generateConvolutionFloatParams<element::Type_t::bf16>(),
        generateConvolutionI8Params<element::Type_t::i8>()};
    std::vector<ConvolutionParams> combinedParams;

    for (const auto& params : convolutionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_With_Hardcoded_Refs,
                         ReferenceConvolutionLayerTest,
                         testing::ValuesIn(generateConvolutionCombinedParams()),
                         ReferenceConvolutionLayerTest::getTestCaseName);

}  // namespace
