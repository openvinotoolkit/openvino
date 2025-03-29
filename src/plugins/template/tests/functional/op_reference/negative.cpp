// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/negative.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct NegativeParams {
    template <class IT>
    NegativeParams(const PartialShape& shape,
                   const element::Type& iType,
                   const std::vector<IT>& iValues,
                   const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceNegativeLayerTest : public testing::TestWithParam<NegativeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NegativeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto negative = std::make_shared<op::v0::Negative>(in);
        return std::make_shared<Model>(NodeVector{negative}, ParameterVector{in});
    }
};

TEST_P(ReferenceNegativeLayerTest, NegativeWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<NegativeParams> generateParamsForNegativeFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<NegativeParams> params{NegativeParams(ov::PartialShape{6},
                                                      IN_ET,
                                                      std::vector<T>{1, -2, 0, -4.75f, 8.75f, -8.75f},
                                                      std::vector<T>{-1, 2, 0, 4.75f, -8.75f, 8.75f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<NegativeParams> generateParamsForNegativeInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<NegativeParams> params{NegativeParams(ov::PartialShape{10},
                                                      IN_ET,
                                                      std::vector<T>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1},
                                                      std::vector<T>{-1, -8, 8, -17, 2, -1, -8, 8, -17, 1})};
    return params;
}

std::vector<NegativeParams> generateCombinedParamsForNegative() {
    const std::vector<std::vector<NegativeParams>> allTypeParams{
        generateParamsForNegativeFloat<element::Type_t::f32>(),
        generateParamsForNegativeFloat<element::Type_t::f16>(),
        generateParamsForNegativeFloat<element::Type_t::bf16>(),
        generateParamsForNegativeInt<element::Type_t::i64>(),
        generateParamsForNegativeInt<element::Type_t::i32>()};

    std::vector<NegativeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Negative_With_Hardcoded_Refs,
                         ReferenceNegativeLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForNegative()),
                         ReferenceNegativeLayerTest::getTestCaseName);
}  // namespace
