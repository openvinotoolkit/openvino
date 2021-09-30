// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/convolution.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConvolutionParams {
    template <class IT>
    ConvolutionParams(const PartialShape& inputShape, const PartialShape& filterShape, const PartialShape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues, const std::vector<IT>& filterValues, const std::vector<IT>& oValues,
                      const Strides& strides, const CoordinateDiff& padBegin, const CoordinateDiff& padEnd, const Strides& dialations)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          filterData(CreateTensor(iType, filterValues)),
          refData(CreateTensor(iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations) {}

    template <class IT>
    ConvolutionParams(const PartialShape& inputShape, const PartialShape& filterShape, const PartialShape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues, const std::vector<IT>& filterValues, const std::vector<IT>& oValues,
                      const Strides& strides, const CoordinateDiff& padBegin, const CoordinateDiff& padEnd, const Strides& dialations,
                      const bool convolutionOutlining)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          filterData(CreateTensor(iType, filterValues)),
          refData(CreateTensor(iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations),
          convolutionOutlining(convolutionOutlining) {}

    PartialShape inputShape;
    PartialShape filterShape;
    PartialShape outputShape;
    ov::element::Type inType;
    ov::element::Type filterType;
    ov::element::Type outType;
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor filterData;
    ov::runtime::Tensor refData;
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
    static std::shared_ptr<Function> CreateFunction(const ConvolutionParams& params) {
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
            return std::make_shared<ov::Function>(NodeVector {Convolution2}, ParameterVector {in, filter});
        } else {
            return std::make_shared<ov::Function>(NodeVector {Convolution}, ParameterVector {in, filter});
        }
    }
};

TEST_P(ReferenceConvolutionLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ConvolutionParams> generateConvolutionFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ConvolutionParams> convolutionParams {
// --------------------- 1D convolution ------------------------------------------
// clang-format off
        ConvolutionParams(PartialShape {1, 1, 6},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{5.0f, 6.0f, 7.0f, 2.0f},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 3.0f, 0.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{3.0f, 5.0f, 6.0f, 6.0f},
                          {1},
                          {1},
                          {1},
                          {1}),
        ConvolutionParams(PartialShape {1, 1, 5},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 3.0f, 0.0f, 1.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{5.0f, 7.0f},
                          {2},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(PartialShape {1, 1, 7},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 3},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{3.0f, 8.0f, 9.0f},
                          {1},
                          {0},
                          {0},
                          {2}),
        ConvolutionParams(PartialShape {1, 1, 7},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{3.0f, 3.0f, 9.0f, 2.0f},
                          {2},
                          {2},
                          {2},
                          {2}),
        ConvolutionParams(PartialShape {1, 2, 4},
                          PartialShape {1, 2, 3},
                          PartialShape {1, 1, 2},
                          IN_ET,
                          std::vector<T>{
                                        // channel 1
                                        1.0f, 3.0f, 2.0f, 1.0f,
                                        // channel 2
                                        2.0f, 2.0f, 3.0f, 1.0f},
                          std::vector<T>{
                                        // channel 1
                                        2.0f, 0.0f, 1.0f,
                                        // channel 2
                                        1.0f, 0.0f, 2.0f},
                          std::vector<T>{12.0f, 11.0f},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(PartialShape {1, 1, 4},
                          PartialShape {2, 1, 3},
                          PartialShape {1, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 2.0f, 1.0f},
                          std::vector<T>{
                                        // filter 1
                                        2.0f, 0.0f, 1.0f,
                                        // filter 2
                                        1.0f, 0.0f, 2.0f},
                          std::vector<T>{
                                        // channel 1
                                        4.0f, 7.0f,
                                        // channel 2
                                        5.0f, 5.0f},
                          {1},
                          {0},
                          {0},
                          {1}),
        ConvolutionParams(PartialShape {2, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {2, 1, 2},
                          IN_ET,
                          std::vector<T>{
                                        // batch 1
                                        1.0f, 3.0f, 2.0f, 1.0f,
                                        // batch 2
                                        2.0f, 2.0f, 3.0f, 1.0f},
                          std::vector<T>{2.0f, 0.0f, 1.0f},
                          std::vector<T>{
                                        // batch 1
                                        4.0f, 7.0f,
                                        // batch 2
                                        7.0f, 5.0f},
                          {1},
                          {0},
                          {0},
                          {1}),
// --------------------- 2D convolution ------------------------------------------
        ConvolutionParams(PartialShape {1, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f},
                          std::vector<T>{1.0f, 2.0f, 3.0f,
                                         0.0f, 1.0f, 0.0f,
                                         3.0f, 2.0f, 1.0f},
                          std::vector<T>{47.0f, 69.0f,
                                         70.0f, 48.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(PartialShape {1, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f},
                          std::vector<T>{1.0f, 2.0f, 3.0f,
                                         0.0f, 1.0f, 0.0f,
                                         2.0f, 1.0f, 2.0f},
                          std::vector<T>{18.0f, 28.0f, 20.0f, 14.0f,
                                        28.0f, 47.0f, 67.0f, 40.0f,
                                         51.0f, 60.0f, 40.0f, 23.0f,
                                         24.0f, 34.0f, 44.0f, 24.0f},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1}),
        ConvolutionParams(PartialShape {1, 1, 5, 5},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f, 0.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f, 0.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f},
                          std::vector<T>{1.0f, 2.0f, 3.0f,
                                         1.0f, 1.0f, 1.0f,
                                         3.0f, 2.0f, 1.0f},
                          std::vector<T>{57.0f, 94.0f,
                                         66.0f, 102.0f},
                          {2, 2},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(PartialShape {1, 1, 7, 7},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f},
                          std::vector<T>{1.0f, 2.0f, 3.0f,
                                         1.0f, 1.0f, 0.0f,
                                         3.0f, 1.0f, 2.0f},
                          std::vector<T>{78.0f, 106.0f, 134.0f,
                                         44.0f, 16.0f, -12.0f,
                                         80.0f, 84.0f, 88.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {2, 2}),
        ConvolutionParams(PartialShape {1, 1, 7, 7},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f},
                          std::vector<T>{1.0f, 2.0f, 3.0f,
                                         1.0f, 1.0f, 0.0f,
                                         3.0f, 1.0f, 2.0f},
                          std::vector<T>{15.0f, 38.0f, 70.0f, 66.0f,
                                         33.0f, 78.0f, 134.0f, 103.0f,
                                         40.0f, 80.0f, 88.0f, 58.0f,
                                         30.0f, 56.0f, 72.0f, 34.0f},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {2, 2}),
        ConvolutionParams(PartialShape {1, 2, 4, 4},
                          PartialShape {1, 2, 3, 3},
                          PartialShape {1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                         // channel 1
                                         1.0f, 3.0f, 5.0f, 7.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f,
                                         // channel 2
                                         -1.0f, 3.0f, -5.0f, 7.0f,
                                         7.0f, -5.0f, 3.0f, -1.0f,
                                         -2.0f, 4.0f, -6.0f, 8.0f,
                                         8.0f, -6.0f, 4.0f, -2.0f},
                          std::vector<T>{
                                         // channel 1
                                         5.0f, 3.0f, 5.0f,
                                         1.0f, 3.0f, 1.0f,
                                         4.0f, 2.0f, 4.0f,
                                         // channel 2
                                         -5.0f, 3.0f, 5.0f,
                                         1.0f, -3.0f, 1.0f,
                                         4.0f, 2.0f, -4.0f},
                          std::vector<T>{142.0f, 102.0f,
                                         94.0f, 160.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(PartialShape {1, 1, 4, 4},
                          PartialShape {2, 1, 3, 3},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 3.0f, 5.0f, 7.0f,
                                         7.0f, 5.0f, 3.0f, 1.0f,
                                         2.0f, 4.0f, 6.0f, 8.0f,
                                         8.0f, 6.0f, 4.0f, 2.0f},
                          std::vector<T>{
                                         // channel 1
                                         5.0f, 3.0f, 5.0f,
                                         1.0f, 3.0f, 1.0f,
                                         4.0f, 2.0f, 4.0f,
                                         // channel 2
                                         -5.0f, 3.0f, 5.0f,
                                         1.0f, -3.0f, 1.0f,
                                         4.0f, 2.0f, -4.0f},
                          std::vector<T>{
                                         // channel 1
                                         104.0f, 140.0f,
                                         145.0f, 109.0f,
                                         // channel 2
                                         16.0f, 28.0f,
                                         19.0f, 7.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(PartialShape {2, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {2, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                         // batch 1
                                         1.0f, 3.0f, 2.0f, 1.0f,
                                         1.0f, 3.0f, 3.0f, 1.0f,
                                         2.0f, 1.0f, 1.0f, 3.0f,
                                         3.0f, 2.0f, 3.0f, 3.0f,
                                         // batch 2
                                         -1.0f, 3.0f, 2.0f, -1.0f,
                                         1.0f, 3.0f, -3.0f, 1.0f,
                                         -2.0f, -1.0f, 1.0f, 3.0f,
                                         3.0f, 2.0f, 3.0f, -3.0f},
                          std::vector<T>{-5.0f, 3.0f, 5.0f,
                                         1.0f, -3.0f, 1.0f,
                                         4.0f, 2.0f, -4.0f},
                          std::vector<T>{
                                         // batch 1
                                         15.0f, -15.0f,
                                         23.0f, 2.0f,
                                         // batch 2
                                         -1.0f, -15.0f,
                                         -5.0f, 6.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
// --------------------- 3D convolution ------------------------------------------
        ConvolutionParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f},
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f},
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // depth: 1
                                     16.0f, 28.0f, 26.0f, 16.0f,
                                     32.0f, 46.0f, 44.0f, 20.0f,
                                     40.0f, 62.0f, 52.0f, 34.0f,
                                     20.0f, 18.0f, 30.0f, 20.0f,
                                     // depth: 2
                                     24.0f, 42.0f, 39.0f, 24.0f,
                                     48.0f, 69.0f, 66.0f, 30.0f,
                                     60.0f, 93.0f, 78.0f, 51.0f,
                                     30.0f, 27.0f, 45.0f, 30.0f,
                                     // depth: 3
                                     24.0f, 42.0f, 39.0f, 24.0f,
                                     48.0f, 69.0f, 66.0f, 30.0f,
                                     60.0f, 93.0f, 78.0f, 51.0f,
                                     30.0f, 27.0f, 45.0f, 30.0f,
                                     // depth: 4
                                     16.0f, 28.0f, 26.0f, 16.0f,
                                     32.0f, 46.0f, 44.0f, 20.0f,
                                     40.0f, 62.0f, 52.0f, 34.0f,
                                     20.0f, 18.0f, 30.0f, 20.0f},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1}),
        ConvolutionParams(PartialShape {1, 1, 5, 5, 5},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    // depth: 5
                                    1.0f, 3.0f, 2.0f, 1.0f, 2.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 2.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 2.0f},
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // depth: 1
                                     69.0f, 60.0f,
                                     69.0f, 87.0f,
                                     // depth: 2
                                     69.0f, 60.0f,
                                     69.0f, 87.0f},
                          {2, 2, 2},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(PartialShape {1, 1, 7, 7, 7},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 5
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 6
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    // depth: 7
                                    1.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f, 1.0f, 2.0f, 3.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f, 3.0f},
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // depth: 1
                                     10.0f, 18.0f, 20.0f, 16.0f,
                                     38.0f, 40.0f, 54.0f, 30.0f,
                                     38.0f, 42.0f, 52.0f, 30.0f,
                                     36.0f, 30.0f, 30.0f, 20.0f,
                                     // depth: 2
                                     15.0f, 27.0f, 30.0f, 24.0f,
                                     57.0f, 60.0f, 81.0f, 45.0f,
                                     57.0f, 63.0f, 78.0f, 45.0f,
                                     54.0f, 45.0f, 45.0f, 30.0f,
                                     // depth: 3
                                     15.0f, 27.0f, 30.0f, 24.0f,
                                     57.0f, 60.0f, 81.0f, 45.0f,
                                     57.0f, 63.0f, 78.0f, 45.0f,
                                     54.0f, 45.0f, 45.0f, 30.0f,
                                     // depth: 4
                                     10.0f, 18.0f, 20.0f, 16.0f,
                                     38.0f, 40.0f, 54.0f, 30.0f,
                                     38.0f, 42.0f, 52.0f, 30.0f,
                                     36.0f, 30.0f, 30.0f, 20.0f},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2}),
        ConvolutionParams(PartialShape {1, 2, 4, 4, 4},
                          PartialShape {1, 2, 3, 3, 3},
                          PartialShape {1, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // -- channel 2 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // -- channel 2 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // depth: 1
                                     138.0f, 132.0f,
                                     186.0f, 156.0f,
                                     // depth: 2
                                     138.0f, 132.0f,
                                     186.0f, 156.0f},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {2, 1, 3, 3, 3},
                          PartialShape {1, 2, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f},
                          std::vector<T>{
                                    // -- filter 1 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // -- filter 2 --
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // -- out 1 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // -- out 2 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
        ConvolutionParams(PartialShape {2, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {2, 1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // -- batch 1 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // -- batch 2 --
                                    // depth: 1
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 2
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 3
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f,
                                    // depth: 4
                                    1.0f, 3.0f, 2.0f, 1.0f,
                                    1.0f, 3.0f, 3.0f, 1.0f,
                                    2.0f, 1.0f, 1.0f, 3.0f,
                                    3.0f, 2.0f, 3.0f, 3.0f},
                          std::vector<T>{
                                    // depth: 1
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 2
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f,
                                    // depth: 3
                                    1.0f, 2.0f, 3.0f,
                                    0.0f, 1.0f, 0.0f,
                                    2.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                     // -- batch 1 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // -- batch 2 --
                                     // depth: 1
                                     69.0f, 66.0f,
                                     93.0f, 78.0f,
                                     // depth: 2
                                     69.0f, 66.0f,
                                     93.0f, 78.0f},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1}),
// ----------------------  other tests ------------------------------------------
        ConvolutionParams(PartialShape {1, 2, 2, 2},
                          PartialShape {2, 2, 1, 1},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                          std::vector<T>{1.0f, 1.0f, 1.0f, 1.0f},
                          std::vector<T>{4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          true),
        ConvolutionParams(PartialShape {1, 2, 2, 2},
                          PartialShape {2, 2, 1, 1},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                          std::vector<T>{3.0f, 3.0f, 3.0f, 3.0f},
                          std::vector<T>{18.0f, 24.0f, 30.0f, 36.0f, 18.0f, 24.0f, 30.0f, 36.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        ConvolutionParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 1, 1},
                          PartialShape {1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{1.0f, 2.0f, 3.0f, 4.0f},
                          std::vector<T>{2.0f},
                          std::vector<T>{
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 2.0f, 4.0f, 0.0f, 0.0f,
                                  0.0f, 6.0f, 8.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                          {1, 1},
                          {1, 1},
                          {2, 2},
                          {1, 1})
    };
    return convolutionParams;
}

std::vector<ConvolutionParams> generateConvolutionCombinedParams() {
    const std::vector<std::vector<ConvolutionParams>> convolutionTypeParams {
        generateConvolutionFloatParams<element::Type_t::f32>(),
        generateConvolutionFloatParams<element::Type_t::f16>(),
        generateConvolutionFloatParams<element::Type_t::bf16>()
        };
    std::vector<ConvolutionParams> combinedParams;

    for (const auto& params : convolutionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Convolution_With_Hardcoded_Refs, ReferenceConvolutionLayerTest,
    testing::ValuesIn(generateConvolutionCombinedParams()), ReferenceConvolutionLayerTest::getTestCaseName);

} // namespace