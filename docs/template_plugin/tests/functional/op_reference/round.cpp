// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace reference_tests;
using namespace InferenceEngine;

namespace {

struct RoundParams {
    template <class IT>
    RoundParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type, const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_TO_EVEN);
        return std::make_shared<Function>(NodeVector{round}, ParameterVector{in});
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type, const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto round = std::make_shared<op::v5::Round>(in, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
        return std::make_shared<Function>(NodeVector{round}, ParameterVector{in});
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

    std::vector<RoundParams> roundParams{
        RoundParams(ngraph::PartialShape{5},
                    IN_ET,
                    std::vector<T>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
                    std::vector<T>{1.0f, 2.0f, 2.0f, 2.0f, -4.0f})
    };
    return roundParams;
}

std::vector<RoundParams> generateCombinedParamsForRoundHalfToEven() {
    const std::vector<std::vector<RoundParams>> roundTypeParams{
        generateParamsForRoundHalfToEven<element::Type_t::f32>(),
        generateParamsForRoundHalfToEven<element::Type_t::f16>()
    };

    std::vector<RoundParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRound2D() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> roundParams{
        RoundParams(ngraph::PartialShape{15},
                    IN_ET,
                    std::vector<T>{0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f, -1.5f, -1.9f, -2.2f, -2.5f, -2.8f},
                    std::vector<T>{0.f, 0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, -1.f, -2.f, -2.f, -2.f, -2.f, -3.f})};
    return roundParams;
}

std::vector<RoundParams> generateCombinedParamsForRound2D() {
    const std::vector<std::vector<RoundParams>> roundTypeParams{
        generateParamsForRound2D<element::Type_t::f32>(),
        generateParamsForRound2D<element::Type_t::f16>()
    };

    std::vector<RoundParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> roundParams{
        RoundParams(ngraph::PartialShape{3},
                    IN_ET,
                    std::vector<T>{0, 1, 0x4000000000000001},
                    std::vector<T>{0, 1, 0x4000000000000001})};
    return roundParams;
}

std::vector<RoundParams> generateCombinedParamsForRoundInt64() {
    const std::vector<std::vector<RoundParams>> roundTypeParams{
        generateParamsForRoundInt64<element::Type_t::i64>()
    };

    std::vector<RoundParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<RoundParams> generateParamsForRoundAwayFromZero() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RoundParams> roundParams{
        RoundParams(ngraph::PartialShape{5},
                    IN_ET,
                    std::vector<T>{0.9f, 2.5f, 2.3f, 1.5f, -4.5f},
                    std::vector<T>{1.0f, 3.0f, 2.0f, 2.0f, -5.0f})};
    return roundParams;
}

std::vector<RoundParams> generateCombinedParamsForRoundAwayFromZero() {
    const std::vector<std::vector<RoundParams>> roundTypeParams{
        generateParamsForRoundAwayFromZero<element::Type_t::f32>(),
        generateParamsForRoundAwayFromZero<element::Type_t::f16>()
    };

    std::vector<RoundParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Half_To_Even_With_Hardcoded_Refs, 
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForRoundHalfToEven()), 
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_2D_With_Hardcoded_Refs,
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForRound2D()),
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Int64_With_Hardcoded_Refs,
    ReferenceRoundHalfToEvenLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForRoundInt64()),
    ReferenceRoundHalfToEvenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Round_Away_From_Zero_With_Hardcoded_Refs,
    ReferenceRoundHalfAwayFromZeroLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForRoundAwayFromZero()),
    ReferenceRoundHalfAwayFromZeroLayerTest::getTestCaseName);

}  // namespace