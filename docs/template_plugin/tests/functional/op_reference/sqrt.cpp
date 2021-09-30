// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "base_reference_test.hpp"
#include "openvino/op/sqrt.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct SqrtParams {
    template <class IT>
    SqrtParams(const PartialShape& shape,
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
    runtime::Tensor inputData;
    runtime::Tensor refData;
};

class ReferenceSqrtLayerTest : public testing::TestWithParam<SqrtParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SqrtParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto sqrt = std::make_shared<op::v0::Sqrt>(in);
        return std::make_shared<Function>(NodeVector{sqrt}, ParameterVector{in});
    }
};

TEST_P(ReferenceSqrtLayerTest, SqrtWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SqrtParams> generateParamsForSqrtBasic() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SqrtParams> roundParams{
        SqrtParams(ngraph::PartialShape{6},
                   IN_ET,
                   std::vector<T>{16, 4, 81, 100, 10000, 0},
                   std::vector<T>{4, 2, 9, 10, 100, 0})
    };
    return roundParams;
}

std::vector<SqrtParams> generateCombinedParamsForSqrtBasic() {
    const std::vector<std::vector<SqrtParams>> roundTypeParams{
        generateParamsForSqrtBasic<element::Type_t::f32>(),
        generateParamsForSqrtBasic<element::Type_t::f16>(),
        generateParamsForSqrtBasic<element::Type_t::i64>(),
        generateParamsForSqrtBasic<element::Type_t::i32>(),
        generateParamsForSqrtBasic<element::Type_t::u64>(),
        generateParamsForSqrtBasic<element::Type_t::u32>()
    };

    std::vector<SqrtParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<SqrtParams> generateParamsForSqrtNegative() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SqrtParams> roundParams{
        SqrtParams(ngraph::PartialShape{4},
                   IN_ET,
                   std::vector<T>{-1, 4, -81, 100},
                   std::vector<T>{static_cast<T> NAN, 2, static_cast<T> NAN, 10})
    };
    return roundParams;
}

std::vector<SqrtParams> generateCombinedParamsForSqrtNegative() {
    const std::vector<std::vector<SqrtParams>> roundTypeParams{
        generateParamsForSqrtNegative<element::Type_t::f32>(),
        generateParamsForSqrtNegative<element::Type_t::f16>(),
        generateParamsForSqrtNegative<element::Type_t::i64>(),
        generateParamsForSqrtNegative<element::Type_t::i32>()
    };

    std::vector<SqrtParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<SqrtParams> generateParamsForSqrtIntegralFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SqrtParams> roundParams{
        SqrtParams(ngraph::PartialShape{14},
                   IN_ET,
                   std::vector<T>{4, 7, 9, 10, 80, 55, 6.25, 0.9, 23.33, 233, 256, 473.7891, 1024, 111108.88},
                   std::vector<T>{2, 2.6457512, 3, 3.1622777, 8.944272, 7.4161983, 2.5, 0.94868326, 4.830114, 15.264338, 16., 21.766697, 32., 333.33})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<SqrtParams> generateParamsForSqrtIntegralInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SqrtParams> roundParams{
        SqrtParams(ngraph::PartialShape{14},
                   IN_ET,
                   std::vector<T>{4, 7, 9, 10, 80, 55, 6, 1, 23, 233, 256, 474, 1024, 110889},
                   std::vector<T>{2, 3, 3, 3, 9, 7, 2, 1, 5, 15, 16, 22, 32, 333})
    };
    return roundParams;
}

std::vector<SqrtParams> generateCombinedParamsForSqrtIntegral() {
    const std::vector<std::vector<SqrtParams>> roundTypeParams{
        generateParamsForSqrtIntegralFloat<element::Type_t::f32>(),
        generateParamsForSqrtIntegralFloat<element::Type_t::f16>(),
        generateParamsForSqrtIntegralInt<element::Type_t::i64>(),
        generateParamsForSqrtIntegralInt<element::Type_t::i32>(),
        generateParamsForSqrtIntegralInt<element::Type_t::u64>(),
        generateParamsForSqrtIntegralInt<element::Type_t::u32>()
    };

    std::vector<SqrtParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Basic_With_Hardcoded_Refs,
    ReferenceSqrtLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForSqrtBasic()),
    ReferenceSqrtLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Negative_With_Hardcoded_Refs,
    ReferenceSqrtLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForSqrtNegative()),
    ReferenceSqrtLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Sqrt_Integral_With_Hardcoded_Refs,
    ReferenceSqrtLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForSqrtIntegral()),
    ReferenceSqrtLayerTest::getTestCaseName);
}  // namespace
