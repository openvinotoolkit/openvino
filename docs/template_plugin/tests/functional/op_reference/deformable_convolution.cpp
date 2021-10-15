// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/deformable_convolution.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DeformableConvolutionParams {
    template <class IT>
    DeformableConvolutionParams(const PartialShape& inputShape, const PartialShape& filterShape,
                      const PartialShape& offsetShape, const PartialShape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues, const std::vector<IT>& filterValues,
                      const std::vector<IT>& offsetValues, const std::vector<IT>& oValues,
                      const Strides& strides, const CoordinateDiff& padBegin, const CoordinateDiff& padEnd, const Strides& dialations,
                      const int64_t group = 1, const int64_t deformableGroup = 1)
        : inputShape(inputShape),
          filterShape(filterShape),
          offsetShape(offsetShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          offsetType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          filterData(CreateTensor(iType, filterValues)),
          offsetData(CreateTensor(iType, offsetValues)),
          refData(CreateTensor(iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations),
          group(group),
          deformableGroup(deformableGroup) {}

    PartialShape inputShape;
    PartialShape filterShape;
    PartialShape offsetShape;
    PartialShape outputShape;
    ov::element::Type inType;
    ov::element::Type filterType;
    ov::element::Type offsetType;
    ov::element::Type outType;
    ov::runtime::Tensor inputData;
    ov::runtime::Tensor filterData;
    ov::runtime::Tensor offsetData;
    ov::runtime::Tensor refData;
    ov::Strides strides;
    ov::CoordinateDiff padBegin;
    ov::CoordinateDiff padEnd;
    ov::Strides dialations;
    int64_t group;
    int64_t deformableGroup;
};

class ReferenceDeformableConvolutionLayerTest : public testing::TestWithParam<DeformableConvolutionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.offsetData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<DeformableConvolutionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "filterShape=" << param.filterShape << "_";
        result << "filterShape=" << param.offsetShape << "_";
        result << "outputShape=" << param.outputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.filterType << "_";
        result << "oType=" << param.offsetType << "_";
        result << "oType=" << param.outType << "_";
        result << "strides=" << param.strides << "_";
        result << "padBegin=" << param.padBegin << "_";
        result << "padEnd=" << param.padEnd << "_";
        result << "dialations=" << param.dialations << "_";
        result << "group=" << param.group << "_";
        result << "deformableGroup=" << param.deformableGroup;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const DeformableConvolutionParams& params) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto offset = std::make_shared<op::v0::Parameter>(params.offsetType, params.offsetShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.filterType, params.filterShape);
        const auto DeformableConvolution = std::make_shared<op::v1::DeformableConvolution>(in,
                                                                       offset,
                                                                       filter,
                                                                       params.strides,
                                                                       params.padBegin,
                                                                       params.padEnd,
                                                                       params.dialations,
                                                                       auto_pad,
                                                                       params.group,
                                                                       params.deformableGroup);
        return std::make_shared<ov::Function>(NodeVector {DeformableConvolution}, ParameterVector {in, offset, filter});
    }
};

TEST_P(ReferenceDeformableConvolutionLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DeformableConvolutionParams> generateDeformableConvolutionFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<DeformableConvolutionParams> deformableConvolutionParams {
// --------------------- 2D DeformableConvolution ------------------------------------------
        DeformableConvolutionParams(PartialShape {1, 1, 4, 4},
                          PartialShape {1, 1, 2, 2},
                          PartialShape {1, 8, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f,
                                    9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f},
                          std::vector<T>{
                                    1.0f, 2.0f,
                                    -1.0f, -2.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    -12.0f, -12.0f, -12.0f,
                                    -12.0f, -12.0f, -12.0f,
                                    -12.0f, -12.0f, -12.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        DeformableConvolutionParams(PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 2, 2},
                          PartialShape {1, 8, 4, 4},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f,
                                    7.0f, 5.0f, 3.0f,
                                    1.0f, 3.0f, 5.0f},
                          std::vector<T>{
                                    1.0f, 2.0f,
                                    0.0f, 1.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f, 0.0f,
                                    9.0f, 12.0f, 16.0f, 5.0f,
                                    15.0f, 20.0f, 16.0f, 3.0f,
                                    2.0f, 7.0f, 13.0f, 5.0f},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1}),
        DeformableConvolutionParams(PartialShape {1, 1, 5, 5},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 18, 2, 2},
                          PartialShape {1, 1, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f, 7.0f, 9.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f},
                          std::vector<T>{
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 1.0f,
                                    3.0f, 2.0f, 1.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    57.0f, 94.0f,
                                    66.0f, 102.0f},
                          {2, 2},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        DeformableConvolutionParams(PartialShape {1, 1, 7, 7},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 18, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f},
                          std::vector<T>{
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    78.0f, 106.0f, 134.0f,
                                    44.0f, 16.0f, -12.0f,
                                    80.0f, 84.0f, 88.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {2, 2}),
        DeformableConvolutionParams(PartialShape {1, 1, 7, 7},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 18, 4, 4},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f,
                                    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f,
                                    7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f,
                                    8.0f, 6.0f, 4.0f, 2.0f, 0.0f, -2.0f, -4.0f},
                          std::vector<T>{
                                    1.0f, 2.0f, 3.0f,
                                    1.0f, 1.0f, 0.0f,
                                    3.0f, 1.0f, 2.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    15.0f, 38.0f, 70.0f, 66.0f,
                                    33.0f, 78.0f, 134.0f, 103.0f,
                                    40.0f, 80.0f, 88.0f, 58.0f,
                                    30.0f, 56.0f, 72.0f, 34.0f},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {2, 2}),
        DeformableConvolutionParams(PartialShape {1, 2, 4, 4},
                          PartialShape {1, 2, 3, 3},
                          PartialShape {1, 18, 2, 2},
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
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    142.0f, 102.0f,
                                    94.0f, 160.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1}),
        DeformableConvolutionParams(PartialShape {1, 1, 4, 4},
                          PartialShape {2, 1, 3, 3},
                          PartialShape {1, 18, 2, 2},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    1.0f, 3.0f, 5.0f, 7.0f,
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
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
        DeformableConvolutionParams(PartialShape {2, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {2, 18, 2, 2},
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
                          std::vector<T>{
                                   -5.0f, 3.0f, 5.0f,
                                    1.0f, -3.0f, 1.0f,
                                    4.0f, 2.0f, -4.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
        DeformableConvolutionParams(PartialShape {1, 4, 3, 3},
                          PartialShape {2, 2, 2, 2},
                          PartialShape {1, 8, 2, 2},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                     // channel 2
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    // channel 3
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    // channel 4
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f},
                          std::vector<T>{
                                    // filter 1 channel 1
                                    1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    // filter 1 channel 2
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    // filter 2 channel 1
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    // filter 2 channel 2
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    // channel 1
                                    356.0f, 392.0f,
                                    464.0f, 500.0f,
                                    // channel 2
                                    -1004.0f, -1040.0f,
                                    -1112.0f, -1148.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          2),
        DeformableConvolutionParams(PartialShape {1, 8, 3, 3},
                          PartialShape {4, 2, 2, 2},
                          PartialShape {1, 8, 2, 2},
                          PartialShape {1, 4, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f,
                                    // channel 2
                                    10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f,
                                    // channel 3
                                    19.0f, 20.0f, 21.0f,
                                    22.0f, 23.0f, 24.0f,
                                    25.0f, 26.0f, 27.0f,
                                    // channel 4
                                    28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f,
                                    34.0f, 35.0f, 36.0f,
                                    // channel 5
                                    37.0f, 38.0f, 39.0f,
                                    40.0f, 41.0f, 42.0f,
                                    43.0f, 44.0f, 45.0f,
                                    // channel 6
                                    46.0f, 47.0f, 48.0f,
                                    49.0f, 50.0f, 51.0f,
                                    52.0f, 53.0f, 54.0f,
                                    // channel 7
                                    55.0f, 56.0f, 57.0f,
                                    58.0f, 59.0f, 60.0f,
                                    61.0f, 62.0f, 63.0f,
                                    // channel 8
                                    64.0f, 65.0f, 66.0f,
                                    67.0f, 68.0f, 69.0f,
                                    70.0f, 71.0f, 72.0f},
                          std::vector<T>{
                                    // filter 1 channel 1
                                    1.0f, 2.0f,
                                    3.0f, 4.0f,
                                    // filter 1 channel 2
                                    5.0f, 6.0f,
                                    7.0f, 8.0f,
                                    // filter 2 channel 1
                                    9.0f, 10.0f,
                                    11.0f, 12.0f,
                                    // filter 2 channel 2
                                    13.0f, 14.0f,
                                    15.0f, 16.0f,
                                    // filter 3 channel 1
                                    -1.0f, -2.0f,
                                    -3.0f, -4.0f,
                                    // filter 3 channel 2
                                    -5.0f, -6.0f,
                                    -7.0f, -8.0f,
                                    // filter 4 channel 1
                                    -9.0f, -10.0f,
                                    -11.0f, -12.0f,
                                    // filter 4 channel 2
                                    -13.0f, -14.0f,
                                    -15.0f, -16.0f},
                          std::vector<T>{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0},
                          std::vector<T>{
                                    // channel 1
                                    356.0f, 392.0f,
                                    464.0f, 500.0f,
                                    // channel 2
                                    2636.0f, 2736.0f,
                                    2936.0f, 3036.0f,
                                    // channel 3
                                    -1652.0f, -1688.0f,
                                    -1760.0f, -1796.0f,
                                    // channel 4
                                    -6236.0f, -6336.0f,
                                    -6536.0f, -6636.0f},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          4),
    };
    return deformableConvolutionParams;
}
// clang-format on

std::vector<DeformableConvolutionParams> generateDeformableConvolutionCombinedParams() {
    const std::vector<std::vector<DeformableConvolutionParams>> deformableConvolutionTypeParams {
        generateDeformableConvolutionFloatParams<element::Type_t::f64>(),
        generateDeformableConvolutionFloatParams<element::Type_t::f32>(),
        generateDeformableConvolutionFloatParams<element::Type_t::f16>(),
        generateDeformableConvolutionFloatParams<element::Type_t::bf16>()
        };
    std::vector<DeformableConvolutionParams> combinedParams;

    for (const auto& params : deformableConvolutionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DeformableConvolution_With_Hardcoded_Refs, ReferenceDeformableConvolutionLayerTest,
    testing::ValuesIn(generateDeformableConvolutionCombinedParams()), ReferenceDeformableConvolutionLayerTest::getTestCaseName);

} // namespace