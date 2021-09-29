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

struct AbsParams {
    template <class IT>
    AbsParams(const PartialShape& shape,
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

class ReferenceAbsLayerTest : public testing::TestWithParam<AbsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AbsParams>& obj) {
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
        const auto log = std::make_shared<op::Abs>(in);
        return std::make_shared<Function>(NodeVector{log}, ParameterVector{in});
    }
};

TEST_P(ReferenceAbsLayerTest, AbsWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> roundParams{
        AbsParams(ngraph::PartialShape{4},
                  IN_ET,
                  std::vector<T>{1.f, -2.f, 0.f, -4.75f},
                  std::vector<T>{1.f, 2.f, 0.f, 4.75f})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> roundParams{
        AbsParams(ngraph::PartialShape{4},
                  IN_ET,
                  std::vector<T>{1, -2, 0, -4},
                  std::vector<T>{1, 2, 0, 4})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<AbsParams> generateParamsForAbsUInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AbsParams> roundParams{
        AbsParams(ngraph::PartialShape{4},
                  IN_ET,
                  std::vector<T>{1, 2, 0, 4},
                  std::vector<T>{1, 2, 0, 4})
    };
    return roundParams;
}

std::vector<AbsParams> generateCombinedParamsForAbs() {
    const std::vector<std::vector<AbsParams>> roundTypeParams{
        generateParamsForAbsFloat<element::Type_t::f32>(),
        generateParamsForAbsFloat<element::Type_t::f16>(),
        generateParamsForAbsInt<element::Type_t::i64>(),
        generateParamsForAbsInt<element::Type_t::i32>(),
        generateParamsForAbsUInt<element::Type_t::u64>(),
        generateParamsForAbsUInt<element::Type_t::u32>()
    };

    std::vector<AbsParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Abs_With_Hardcoded_Refs, 
    ReferenceAbsLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForAbs()),
    ReferenceAbsLayerTest::getTestCaseName);

}  // namespace
