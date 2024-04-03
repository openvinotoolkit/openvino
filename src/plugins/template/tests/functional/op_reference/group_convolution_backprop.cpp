// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GroupConvolutionBackpropDataParams {
    template <class IT>
    GroupConvolutionBackpropDataParams(const Shape& inputShape,
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
                                       const CoordinateDiff& outPadding = {})
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
          outPadding(outPadding) {}

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
    ov::CoordinateDiff outPadding;
};

struct GroupConvolutionBackpropDataOutShapeParams {
    template <class IT>
    GroupConvolutionBackpropDataOutShapeParams(const Shape& inputShape,
                                               const Shape& filterShape,
                                               const Shape& outputShape,
                                               const element::Type& iType,
                                               const std::vector<IT>& iValues,
                                               const std::vector<IT>& filterValues,
                                               const std::vector<IT>& oValues,
                                               const Strides& strides,
                                               const Strides& dialations,
                                               const Shape& constantOutputShape,
                                               const std::vector<int64_t>& constantOutputShapeData)
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
          dialations(dialations),
          constantOutputShape(constantOutputShape),
          constantOutputShapeData(constantOutputShapeData) {}

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
    ov::Strides dialations;
    Shape constantOutputShape;
    std::vector<int64_t> constantOutputShapeData;
};

class ReferenceGroupConvolutionBackpropDataLayerTest
    : public testing::TestWithParam<GroupConvolutionBackpropDataParams>,
      public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GroupConvolutionBackpropDataParams>& obj) {
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
        if (param.outPadding.size() != 0)
            result << "_outPadding=" << param.outPadding;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GroupConvolutionBackpropDataParams& params) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.inType, params.filterShape);
        if (params.outPadding.size() != 0) {
            const auto GroupConvolutionBackpropData =
                std::make_shared<op::v1::GroupConvolutionBackpropData>(in,
                                                                       filter,
                                                                       params.strides,
                                                                       params.padBegin,
                                                                       params.padEnd,
                                                                       params.dialations,
                                                                       auto_pad,
                                                                       params.outPadding);
            return std::make_shared<ov::Model>(NodeVector{GroupConvolutionBackpropData}, ParameterVector{in, filter});
        } else {
            const auto GroupConvolutionBackpropData =
                std::make_shared<op::v1::GroupConvolutionBackpropData>(in,
                                                                       filter,
                                                                       params.strides,
                                                                       params.padBegin,
                                                                       params.padEnd,
                                                                       params.dialations,
                                                                       auto_pad);
            return std::make_shared<ov::Model>(NodeVector{GroupConvolutionBackpropData}, ParameterVector{in, filter});
        }
    }
};

class ReferenceGroupConvolutionBackpropDataLayerOutShapeTest
    : public testing::TestWithParam<GroupConvolutionBackpropDataOutShapeParams>,
      public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GroupConvolutionBackpropDataOutShapeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "filterShape=" << param.filterShape << "_";
        result << "outputShape=" << param.outputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "strides=" << param.strides << "_";
        result << "dialations=" << param.dialations << "_";
        result << "constantOutputShape=" << param.constantOutputShape;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GroupConvolutionBackpropDataOutShapeParams& params) {
        const op::PadType auto_pad{op::PadType::SAME_UPPER};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.inType, params.filterShape);
        auto output_shape = std::make_shared<op::v0::Constant>(element::i64,
                                                               params.constantOutputShape,
                                                               params.constantOutputShapeData);
        const auto GroupConvolutionBackpropData =
            std::make_shared<op::v1::GroupConvolutionBackpropData>(in,
                                                                   filter,
                                                                   output_shape,
                                                                   params.strides,
                                                                   params.dialations,
                                                                   auto_pad);
        return std::make_shared<ov::Model>(NodeVector{GroupConvolutionBackpropData}, ParameterVector{in, filter});
    }
};

TEST_P(ReferenceGroupConvolutionBackpropDataLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceGroupConvolutionBackpropDataLayerOutShapeTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionBackpropDataParams> generateGroupConvolutionBackpropDataFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionBackpropDataParams> groupConvolutionBackpropDataParams{
        // --------------------- 1D GroupConvolutionBackpropData ------------------------------------------
        GroupConvolutionBackpropDataParams(Shape{1, 1, 4},
                                           Shape{1, 1, 1, 3},
                                           Shape{1, 1, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0},
                                           std::vector<T>{2, 0, 1},
                                           std::vector<T>{2, 6, 7, 3, 3, 0},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(Shape{1, 2, 4},
                                           Shape{2, 1, 1, 3},
                                           Shape{1, 2, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0, 1, 2, 1, 3},
                                           std::vector<T>{1, 0, 3, 3, 0, 1},
                                           std::vector<T>{1, 3, 6, 9, 9, 0, 3, 6, 4, 11, 1, 3},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(Shape{1, 4, 4},
                                           Shape{2, 2, 1, 3},
                                           Shape{1, 2, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0, 1, 2, -1, -3, -3, 0, 1, 2, 0, -2, 3, -1},
                                           std::vector<T>{1, 0, 3, 3, 0, 1, -3, 0, 1, 3, 2, -1},
                                           std::vector<T>{4, 9, 4, 2, 8, -3, 9, -6, -1, -1, -4, 3},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(
            Shape{2, 2, 4},
            Shape{2, 1, 1, 3},
            Shape{2, 2, 6},
            IN_ET,
            std::vector<T>{// -- batch 1 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2,
                           // -- batch 2 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2},
            std::vector<T>{1, 0, 3, 3, 0, 1},
            std::vector<T>{1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2, 1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2},
            {1},
            {0},
            {0},
            {1}),
        GroupConvolutionBackpropDataParams(
            Shape{1, 1, 3, 3},
            Shape{1, 1, 1, 3, 3},
            Shape{1, 1, 6, 6},
            IN_ET,
            std::vector<T>{0.16857791f,
                           -0.15161794f,
                           0.08540368f,
                           0.1820628f,
                           -0.21746576f,
                           0.08245695f,
                           0.1431433f,
                           -0.43156421f,
                           0.30591947f},
            std::vector<T>{-0.06230065f,
                           0.37932432f,
                           -0.25388849f,
                           0.33878803f,
                           0.43709868f,
                           -0.22477469f,
                           0.04118127f,
                           -0.44696793f,
                           0.06373066f},
            std::vector<T>{0.07368518f,  -0.08925839f, -0.06627201f, 0.06301362f,  0.03732984f,  -0.01919658f,
                           -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f,  -0.00689478f, -0.01549204f,
                           0.07957941f,  -0.11459791f, -0.09505399f, 0.07681622f,  0.03604182f,  -0.01853423f,
                           -0.0270785f,  -0.00680824f, -0.06650258f, 0.08004665f,  0.07918708f,  -0.0724144f,
                           0.06256775f,  -0.17838378f, -0.18863615f, 0.20064656f,  0.133717f,    -0.06876295f,
                           -0.06398046f, -0.00864975f, 0.19289537f,  -0.01490572f, -0.13673618f, 0.01949645f},
            {2, 2},
            {1, 1},
            {1, 1},
            {1, 1},
            {1, 1}),
    };
    return groupConvolutionBackpropDataParams;
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionBackpropDataParams> generateGroupConvolutionBackpropDataIntParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionBackpropDataParams> groupConvolutionBackpropDataParams{
        // --------------------- 1D GroupConvolutionBackpropData ------------------------------------------
        GroupConvolutionBackpropDataParams(Shape{1, 1, 4},
                                           Shape{1, 1, 1, 3},
                                           Shape{1, 1, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0},
                                           std::vector<T>{2, 0, 1},
                                           std::vector<T>{2, 6, 7, 3, 3, 0},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(Shape{1, 2, 4},
                                           Shape{2, 1, 1, 3},
                                           Shape{1, 2, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0, 1, 2, 1, 3},
                                           std::vector<T>{1, 0, 3, 3, 0, 1},
                                           std::vector<T>{1, 3, 6, 9, 9, 0, 3, 6, 4, 11, 1, 3},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(Shape{1, 4, 4},
                                           Shape{2, 2, 1, 3},
                                           Shape{1, 2, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0, 1, 2, -1, -3, -3, 0, 1, 2, 0, -2, 3, -1},
                                           std::vector<T>{1, 0, 3, 3, 0, 1, -3, 0, 1, 3, 2, -1},
                                           std::vector<T>{4, 9, 4, 2, 8, -3, 9, -6, -1, -1, -4, 3},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(
            Shape{2, 2, 4},
            Shape{2, 1, 1, 3},
            Shape{2, 2, 6},
            IN_ET,
            std::vector<T>{// -- batch 1 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2,
                           // -- batch 2 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2},
            std::vector<T>{1, 0, 3, 3, 0, 1},
            std::vector<T>{1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2, 1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2},
            {1},
            {0},
            {0},
            {1}),
    };
    return groupConvolutionBackpropDataParams;
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionBackpropDataParams> generateGroupConvolutionBackpropDataUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionBackpropDataParams> groupConvolutionBackpropDataParams{
        // --------------------- 1D GroupConvolutionBackpropData ------------------------------------------
        GroupConvolutionBackpropDataParams(Shape{1, 1, 4},
                                           Shape{1, 1, 1, 3},
                                           Shape{1, 1, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0},
                                           std::vector<T>{2, 0, 1},
                                           std::vector<T>{2, 6, 7, 3, 3, 0},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(Shape{1, 2, 4},
                                           Shape{2, 1, 1, 3},
                                           Shape{1, 2, 6},
                                           IN_ET,
                                           std::vector<T>{1, 3, 3, 0, 1, 2, 1, 3},
                                           std::vector<T>{1, 0, 3, 3, 0, 1},
                                           std::vector<T>{1, 3, 6, 9, 9, 0, 3, 6, 4, 11, 1, 3},
                                           {1},
                                           {0},
                                           {0},
                                           {1}),
        GroupConvolutionBackpropDataParams(
            Shape{2, 2, 4},
            Shape{2, 1, 1, 3},
            Shape{2, 2, 6},
            IN_ET,
            std::vector<T>{// -- batch 1 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2,
                           // -- batch 2 --
                           1,
                           3,
                           0,
                           1,
                           1,
                           3,
                           0,
                           2},
            std::vector<T>{1, 0, 3, 3, 0, 1},
            std::vector<T>{1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2, 1, 3, 3, 10, 0, 3, 3, 9, 1, 9, 0, 2},
            {1},
            {0},
            {0},
            {1}),
    };
    return groupConvolutionBackpropDataParams;
}

std::vector<GroupConvolutionBackpropDataParams> generateGroupConvolutionBackpropDataCombinedParams() {
    const std::vector<std::vector<GroupConvolutionBackpropDataParams>> groupConvolutionBackpropDataTypeParams{
        generateGroupConvolutionBackpropDataFloatParams<element::Type_t::f64>(),
        generateGroupConvolutionBackpropDataFloatParams<element::Type_t::f32>(),
        generateGroupConvolutionBackpropDataFloatParams<element::Type_t::f16>(),
        generateGroupConvolutionBackpropDataFloatParams<element::Type_t::bf16>(),
        generateGroupConvolutionBackpropDataIntParams<element::Type_t::i64>(),
        generateGroupConvolutionBackpropDataIntParams<element::Type_t::i32>(),
        generateGroupConvolutionBackpropDataIntParams<element::Type_t::i16>(),
        generateGroupConvolutionBackpropDataIntParams<element::Type_t::i8>(),
        generateGroupConvolutionBackpropDataUintParams<element::Type_t::u64>(),
        generateGroupConvolutionBackpropDataUintParams<element::Type_t::u32>(),
        generateGroupConvolutionBackpropDataUintParams<element::Type_t::u16>(),
        generateGroupConvolutionBackpropDataUintParams<element::Type_t::u8>()};
    std::vector<GroupConvolutionBackpropDataParams> combinedParams;

    for (const auto& params : groupConvolutionBackpropDataTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionBackpropDataOutShapeParams> generateGroupConvolutionBackpropDataOutShapeParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionBackpropDataOutShapeParams> groupConvolutionBackpropDataOutShapeParams{
        // --------------------- 1D GroupConvolutionBackpropData ------------------------------------------
        GroupConvolutionBackpropDataOutShapeParams(Shape{1, 1, 1, 10},
                                                   Shape{1, 1, 1, 1, 5},
                                                   Shape{1, 1, 1, 14},
                                                   IN_ET,
                                                   std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                   std::vector<T>{1, 2, 3, 2, 1},
                                                   std::vector<T>{0, 1, 4, 10, 18, 27, 36, 45, 54, 63, 62, 50, 26, 9},
                                                   {1, 1},
                                                   {1, 1},
                                                   {2},
                                                   {1, 14}),
    };
    return groupConvolutionBackpropDataOutShapeParams;
}

std::vector<GroupConvolutionBackpropDataOutShapeParams> generateGroupConvolutionBackpropDataOutShapeCombinedParams() {
    const std::vector<std::vector<GroupConvolutionBackpropDataOutShapeParams>>
        groupConvolutionBackpropDataOutShapeTypeParams{
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::f64>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::f32>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::f16>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::bf16>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::i64>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::i32>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::i16>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::i8>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::u64>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::u32>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::u16>(),
            generateGroupConvolutionBackpropDataOutShapeParams<element::Type_t::u8>()};
    std::vector<GroupConvolutionBackpropDataOutShapeParams> combinedParams;

    for (const auto& params : groupConvolutionBackpropDataOutShapeTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_With_Hardcoded_Refs,
                         ReferenceGroupConvolutionBackpropDataLayerTest,
                         testing::ValuesIn(generateGroupConvolutionBackpropDataCombinedParams()),
                         ReferenceGroupConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropDataOutShape_With_Hardcoded_Refs,
                         ReferenceGroupConvolutionBackpropDataLayerOutShapeTest,
                         testing::ValuesIn(generateGroupConvolutionBackpropDataOutShapeCombinedParams()),
                         ReferenceGroupConvolutionBackpropDataLayerOutShapeTest::getTestCaseName);

}  // namespace
