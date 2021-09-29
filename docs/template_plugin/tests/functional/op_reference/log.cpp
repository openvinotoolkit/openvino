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

struct LogParams {
    template <class IT>
    LogParams(const PartialShape& shape,
                const element::Type& iType,
                const std::vector<IT>& iValues,
                const std::vector<IT>& oValues)
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type, const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto log = std::make_shared<op::Log>(in);
        return std::make_shared<Function>(NodeVector{log}, ParameterVector{in});
    }
};

TEST_P(ReferenceLogLayerTest, LogWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<LogParams> generateParamsForLog() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<LogParams> roundParams{
        LogParams(ngraph::PartialShape{8},
                  IN_ET,
                  std::vector<T>{0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f},
                  std::vector<T>{-2.07944154f, -1.38629436f, -0.69314718f, 0.00000000f, 0.69314718f, 1.38629436f, 2.07944154f, 2.77258872f})
    };
    return roundParams;
}

std::vector<LogParams> generateCombinedParamsForLog() {
    const std::vector<std::vector<LogParams>> roundTypeParams{
        generateParamsForLog<element::Type_t::f32>(),
        generateParamsForLog<element::Type_t::f16>()
    };

    std::vector<LogParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Log_With_Hardcoded_Refs, 
    ReferenceLogLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForLog()),
    ReferenceLogLayerTest::getTestCaseName);

}  // namespace
