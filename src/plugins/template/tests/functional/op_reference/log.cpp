// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct LogParams {
    template <class IT>
    LogParams(const PartialShape& shape,
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

class ReferenceLogLayerTest : public testing::TestWithParam<LogParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LogParams>& obj) {
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
        const auto log = std::make_shared<op::v0::Log>(in);
        return std::make_shared<Model>(NodeVector{log}, ParameterVector{in});
    }
};

TEST_P(ReferenceLogLayerTest, LogWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<LogParams> generateParamsForLog() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<LogParams> logParams{LogParams(ov::PartialShape{8},
                                               IN_ET,
                                               std::vector<T>{0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f},
                                               std::vector<T>{-2.07944154f,
                                                              -1.38629436f,
                                                              -0.69314718f,
                                                              0.00000000f,
                                                              0.69314718f,
                                                              1.38629436f,
                                                              2.07944154f,
                                                              2.77258872f})};
    return logParams;
}

template <element::Type_t IN_ET>
std::vector<LogParams> generateParamsForLogInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<LogParams> logParams{
        LogParams(ov::PartialShape{4}, IN_ET, std::vector<T>{10, 100, 1000, 10000}, std::vector<T>{2, 4, 6, 9})};
    return logParams;
}

std::vector<LogParams> generateCombinedParamsForLog() {
    const std::vector<std::vector<LogParams>> allTypeParams{generateParamsForLog<element::Type_t::f32>(),
                                                            generateParamsForLog<element::Type_t::f16>(),
                                                            generateParamsForLogInt<element::Type_t::i64>(),
                                                            generateParamsForLogInt<element::Type_t::i32>(),
                                                            generateParamsForLogInt<element::Type_t::u64>(),
                                                            generateParamsForLogInt<element::Type_t::u32>()};

    std::vector<LogParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Log_With_Hardcoded_Refs,
                         ReferenceLogLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForLog()),
                         ReferenceLogLayerTest::getTestCaseName);

}  // namespace
