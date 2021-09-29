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

struct NegativeParams {
    template <class IT>
    NegativeParams(const PartialShape& shape,
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type, const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto negative = std::make_shared<op::Negative>(in);
        return std::make_shared<Function>(NodeVector {negative}, ParameterVector {in});
    }
};

TEST_P(ReferenceNegativeLayerTest, NegativeWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<NegativeParams> generateParamsForNegativeFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<NegativeParams> roundParams{
        NegativeParams(ngraph::PartialShape{6},
                       IN_ET,
                       std::vector<T>{1, -2, 0, -4.75f, 8.75f, -8.75f},
                       std::vector<T>{-1, 2, 0, 4.75f, -8.75f, 8.75f})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<NegativeParams> generateParamsForNegativeInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<NegativeParams> roundParams{
        NegativeParams(ngraph::PartialShape{10},
                       IN_ET,
                       std::vector<T>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1},
                       std::vector<T>{-1, -8, 8, -17, 2, -1, -8, 8, -17, 1})
    };
    return roundParams;
}

std::vector<NegativeParams> generateCombinedParamsForNegative() {
    const std::vector<std::vector<NegativeParams>> roundTypeParams{
        generateParamsForNegativeFloat<element::Type_t::f32>(),
        generateParamsForNegativeFloat<element::Type_t::f16>(),
        generateParamsForNegativeInt<element::Type_t::i64>(),
        generateParamsForNegativeInt<element::Type_t::i32>()
    };

    std::vector<NegativeParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Negative_With_Hardcoded_Refs, 
    ReferenceNegativeLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForNegative()),
    ReferenceNegativeLayerTest::getTestCaseName);
}  // namespace reference_tests
