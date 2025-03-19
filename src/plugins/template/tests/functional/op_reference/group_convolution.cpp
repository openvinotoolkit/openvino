// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/group_conv.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GroupConvolutionParams {
    template <class IT>
    GroupConvolutionParams(const Shape& inputShape,
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
};

class ReferenceGroupConvolutionLayerTest : public testing::TestWithParam<GroupConvolutionParams>,
                                           public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GroupConvolutionParams>& obj) {
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
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GroupConvolutionParams& params) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.inType, params.filterShape);
        const auto GroupConvolution = std::make_shared<op::v1::GroupConvolution>(in,
                                                                                 filter,
                                                                                 params.strides,
                                                                                 params.padBegin,
                                                                                 params.padEnd,
                                                                                 params.dialations,
                                                                                 auto_pad);
        return std::make_shared<ov::Model>(NodeVector{GroupConvolution}, ParameterVector{in, filter});
    }
};

TEST_P(ReferenceGroupConvolutionLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionParams> generateGroupConvolutionParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionParams> groupConvolutionParams{
        // --------------------- 1D GroupConvolution ------------------------------------------
        GroupConvolutionParams(Shape{1, 1, 6},
                               Shape{1, 1, 1, 3},
                               Shape{1, 1, 4},
                               IN_ET,
                               std::vector<T>{1, 3, 3, 0, 1, 2},
                               std::vector<T>{2, 0, 1},
                               std::vector<T>{5, 6, 7, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
        GroupConvolutionParams(Shape{1, 2, 6},
                               Shape{2, 1, 1, 3},
                               Shape{1, 2, 4},
                               IN_ET,
                               std::vector<T>{1, 3, 3, 0, 1, 2, 1, 3, 3, 0, 1, 2},
                               std::vector<T>{1, 0, 3, 3, 0, 1},
                               std::vector<T>{10, 3, 6, 6, 6, 9, 10, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
        GroupConvolutionParams(Shape{1, 2, 6},
                               Shape{2, 2, 1, 3},
                               Shape{1, 4, 4},
                               IN_ET,
                               std::vector<T>{1, 3, 3, 0, 1, 2, -1, -3, -3, 0, 1, 2},
                               std::vector<T>{1, 0, 3, 3, 0, 1, -3, 0, 1, 3, 2, -1},
                               std::vector<T>{10, 3, 6, 6, 6, 9, 10, 2, 0, 9, 10, 2, -6, -15, -10, 0},
                               {1},
                               {0},
                               {0},
                               {1}),
        GroupConvolutionParams(Shape{2, 2, 6},
                               Shape{2, 1, 1, 3},
                               Shape{2, 2, 4},
                               IN_ET,
                               std::vector<T>{// -- batch 1 --
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              // -- batch 2 --
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2},
                               std::vector<T>{1, 0, 3, 3, 0, 1},
                               std::vector<T>{10, 3, 6, 6, 6, 9, 10, 2, 10, 3, 6, 6, 6, 9, 10, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
    };
    return groupConvolutionParams;
}

template <element::Type_t IN_ET>
std::vector<GroupConvolutionParams> generateGroupConvolutionUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<GroupConvolutionParams> groupConvolutionParams{
        // --------------------- 1D GroupConvolution ------------------------------------------
        GroupConvolutionParams(Shape{1, 1, 6},
                               Shape{1, 1, 1, 3},
                               Shape{1, 1, 4},
                               IN_ET,
                               std::vector<T>{1, 3, 3, 0, 1, 2},
                               std::vector<T>{2, 0, 1},
                               std::vector<T>{5, 6, 7, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
        GroupConvolutionParams(Shape{1, 2, 6},
                               Shape{2, 1, 1, 3},
                               Shape{1, 2, 4},
                               IN_ET,
                               std::vector<T>{1, 3, 3, 0, 1, 2, 1, 3, 3, 0, 1, 2},
                               std::vector<T>{1, 0, 3, 3, 0, 1},
                               std::vector<T>{10, 3, 6, 6, 6, 9, 10, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
        GroupConvolutionParams(Shape{2, 2, 6},
                               Shape{2, 1, 1, 3},
                               Shape{2, 2, 4},
                               IN_ET,
                               std::vector<T>{// -- batch 1 --
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              // -- batch 2 --
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2,
                                              1,
                                              3,
                                              3,
                                              0,
                                              1,
                                              2},
                               std::vector<T>{1, 0, 3, 3, 0, 1},
                               std::vector<T>{10, 3, 6, 6, 6, 9, 10, 2, 10, 3, 6, 6, 6, 9, 10, 2},
                               {1},
                               {0},
                               {0},
                               {1}),
    };
    return groupConvolutionParams;
}

std::vector<GroupConvolutionParams> generateGroupConvolutionCombinedParams() {
    const std::vector<std::vector<GroupConvolutionParams>> groupConvolutionTypeParams{
        generateGroupConvolutionParams<element::Type_t::f64>(),
        generateGroupConvolutionParams<element::Type_t::f32>(),
        generateGroupConvolutionParams<element::Type_t::f16>(),
        generateGroupConvolutionParams<element::Type_t::bf16>(),
        generateGroupConvolutionParams<element::Type_t::i64>(),
        generateGroupConvolutionParams<element::Type_t::i32>(),
        generateGroupConvolutionParams<element::Type_t::i16>(),
        generateGroupConvolutionParams<element::Type_t::i8>(),
        generateGroupConvolutionUintParams<element::Type_t::u64>(),
        generateGroupConvolutionUintParams<element::Type_t::u32>(),
        generateGroupConvolutionUintParams<element::Type_t::u16>(),
        generateGroupConvolutionUintParams<element::Type_t::u8>()};
    std::vector<GroupConvolutionParams> combinedParams;

    for (const auto& params : groupConvolutionTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_With_Hardcoded_Refs,
                         ReferenceGroupConvolutionLayerTest,
                         testing::ValuesIn(generateGroupConvolutionCombinedParams()),
                         ReferenceGroupConvolutionLayerTest::getTestCaseName);

}  // namespace
