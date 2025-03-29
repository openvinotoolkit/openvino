// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct RoundParams {
    template <class IT>
    RoundParams(const PartialShape& shape,
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

class ReferenceRoundHalfToEvenLayerTest : public testing::TestWithParam<RoundParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RoundParams>& obj) {
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
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_TO_EVEN);
        return std::make_shared<Model>(NodeVector{round}, ParameterVector{in});
    }
};

class ReferenceRoundHalfAwayFromZeroLayerTest : public testing::TestWithParam<RoundParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RoundParams>& obj) {
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
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        return std::make_shared<Model>(NodeVector{round}, ParameterVector{in});
    }
};

TEST_P(ReferenceRoundHalfToEvenLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceRoundHalfAwayFromZeroLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundHalfToEven() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> params{RoundParams(ov::PartialShape{5},
                                                IN_ET,
                                                std::vector<T>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
                                                std::vector<T>{1.0f, 2.0f, 2.0f, 2.0f, -4.0f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRound2D() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> params{RoundParams(
        ov::PartialShape{15},
        IN_ET,
        std::vector<T>{0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f, -1.5f, -1.9f, -2.2f, -2.5f, -2.8f},
        std::vector<T>{0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> params{RoundParams(ov::PartialShape{3},
                                                IN_ET,
                                                std::vector<T>{0, 1, 0x4000000000000001},
                                                std::vector<T>{0, 1, 0x4000000000000001})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> params{
        RoundParams(ov::PartialShape{3}, IN_ET, std::vector<T>{0, 1, 0x40}, std::vector<T>{0, 1, 0x40})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundAwayFromZero() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> params{RoundParams(ov::PartialShape{5},
                                                IN_ET,
                                                std::vector<T>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
                                                std::vector<T>{1.0f, 3.0f, 2.0f, 2.0f, -5.0f})};
    return params;
}

std::vector<RoundParams> generateCombinedParamsForRoundHalfToEven() {
    const std::vector<std::vector<RoundParams>> allTypeParams{
        generateParamsForRoundHalfToEven<element::Type_t::f32>(),
        generateParamsForRoundHalfToEven<element::Type_t::f16>(),
        generateParamsForRoundHalfToEven<element::Type_t::bf16>(),
    };

    std::vector<RoundParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<RoundParams> generateCombinedParamsForRound2D() {
    const std::vector<std::vector<RoundParams>> allTypeParams{generateParamsForRound2D<element::Type_t::f32>(),
                                                              generateParamsForRound2D<element::Type_t::f16>(),
                                                              generateParamsForRound2D<element::Type_t::bf16>()};

    std::vector<RoundParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<RoundParams> generateCombinedParamsForRoundInt() {
    const std::vector<std::vector<RoundParams>> allTypeParams{generateParamsForRoundInt64<element::Type_t::i64>(),
                                                              generateParamsForRoundInt<element::Type_t::i32>(),
                                                              generateParamsForRoundInt<element::Type_t::i16>(),
                                                              generateParamsForRoundInt<element::Type_t::i8>(),
                                                              generateParamsForRoundInt<element::Type_t::u64>(),
                                                              generateParamsForRoundInt<element::Type_t::u32>(),
                                                              generateParamsForRoundInt<element::Type_t::u16>(),
                                                              generateParamsForRoundInt<element::Type_t::u8>()};

    std::vector<RoundParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<RoundParams> generateCombinedParamsForRoundAwayFromZero() {
    const std::vector<std::vector<RoundParams>> allTypeParams{
        generateParamsForRoundAwayFromZero<element::Type_t::f32>(),
        generateParamsForRoundAwayFromZero<element::Type_t::f16>(),
        generateParamsForRoundAwayFromZero<element::Type_t::bf16>()};

    std::vector<RoundParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Round_Half_To_Even_With_Hardcoded_Refs,
                         ReferenceRoundHalfToEvenLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRoundHalfToEven()),
                         ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Round_2D_With_Hardcoded_Refs,
                         ReferenceRoundHalfToEvenLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRound2D()),
                         ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Round_Int_With_Hardcoded_Refs,
                         ReferenceRoundHalfToEvenLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRoundInt()),
                         ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Round_Away_From_Zero_With_Hardcoded_Refs,
                         ReferenceRoundHalfAwayFromZeroLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRoundAwayFromZero()),
                         ReferenceRoundHalfAwayFromZeroLayerTest::getTestCaseName);

}  // namespace
