// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/subtract.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct SubtractParams {
    template <class IT>
    SubtractParams(const Shape& iShape1,
                   const Shape& iShape2,
                   const Shape& oShape,
                   const element::Type& iType,
                   const std::vector<IT>& iValues1,
                   const std::vector<IT>& iValues2,
                   const std::vector<IT>& oValues)
        : pshape1(iShape1),
          pshape2(iShape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(iShape1, iType, iValues1)),
          inputData2(CreateTensor(iShape2, iType, iValues2)),
          refData(CreateTensor(oShape, iType, oValues)) {}

    Shape pshape1;
    Shape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceSubtractLayerTest : public testing::TestWithParam<SubtractParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SubtractParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape1=" << param.pshape1 << "_";
        result << "iShape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto subtract = std::make_shared<op::v1::Subtract>(in1, in2);

        return std::make_shared<Model>(NodeVector{subtract}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceSubtractLayerTest, SubtractWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SubtractParams> generateParamsForSubtract() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SubtractParams> params{
        SubtractParams(ov::Shape{2, 2},
                       ov::Shape{2, 2},
                       ov::Shape{2, 2},
                       IN_ET,
                       std::vector<T>{2, 4, 8, 16},
                       std::vector<T>{1, 2, 4, 8},
                       std::vector<T>{1, 2, 4, 8}),
        SubtractParams(ov::Shape{3, 2, 1},
                       ov::Shape{1, 6},
                       ov::Shape{3, 2, 6},
                       IN_ET,
                       std::vector<T>{12, 24, 36, 48, 60, 72},
                       std::vector<T>{1, 2, 3, 4, 6, 1},
                       std::vector<T>{11, 10, 9,  8,  6,  11, 23, 22, 21, 20, 18, 23, 35, 34, 33, 32, 30, 35,
                                      47, 46, 45, 44, 42, 47, 59, 58, 57, 56, 54, 59, 71, 70, 69, 68, 66, 71}),
        SubtractParams(ov::Shape{1},
                       ov::Shape{1},
                       ov::Shape{1},
                       IN_ET,
                       std::vector<T>{8},
                       std::vector<T>{2},
                       std::vector<T>{6})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<SubtractParams> generateParamsForSubtractFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SubtractParams> params{SubtractParams(ov::Shape{1},
                                                      ov::Shape{1},
                                                      ov::Shape{1},
                                                      IN_ET,
                                                      std::vector<T>{3.1},
                                                      std::vector<T>{8},
                                                      std::vector<T>{-4.9})};
    return params;
}

std::vector<SubtractParams> generateCombinedParamsForSubtract() {
    const std::vector<std::vector<SubtractParams>> allTypeParams{generateParamsForSubtract<element::Type_t::f32>(),
                                                                 generateParamsForSubtract<element::Type_t::f16>(),
                                                                 generateParamsForSubtract<element::Type_t::bf16>(),
                                                                 generateParamsForSubtract<element::Type_t::i64>(),
                                                                 generateParamsForSubtract<element::Type_t::i32>(),
                                                                 generateParamsForSubtract<element::Type_t::u64>(),
                                                                 generateParamsForSubtract<element::Type_t::u32>()};

    std::vector<SubtractParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<SubtractParams> generateCombinedParamsForSubtractFloat() {
    const std::vector<std::vector<SubtractParams>> allTypeParams{
        generateParamsForSubtractFloat<element::Type_t::f32>(),
        generateParamsForSubtractFloat<element::Type_t::f16>()};

    std::vector<SubtractParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Subtract_With_Hardcoded_Refs,
                         ReferenceSubtractLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForSubtract()),
                         ReferenceSubtractLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Subtract_Float_With_Hardcoded_Refs,
                         ReferenceSubtractLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForSubtractFloat()),
                         ReferenceSubtractLayerTest::getTestCaseName);

}  // namespace
