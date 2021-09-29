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

struct CeilingParams {
    template <class IT>
    CeilingParams(const PartialShape& shape,
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

class ReferenceCeilingLayerTest : public testing::TestWithParam<CeilingParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CeilingParams>& obj) {
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
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto ceiling = std::make_shared<op::Ceiling>(in);
        return std::make_shared<Function>(NodeVector {ceiling}, ParameterVector {in});
    }
};

TEST_P(ReferenceCeilingLayerTest, CeilingWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> roundParams{
        CeilingParams(ngraph::PartialShape{4},
                      IN_ET,
                      std::vector<T>{-2.5f, -2.0f, 0.3f, 4.8f},
                      std::vector<T>{-2.0f, -2.0f, 1.0f, 5.0f})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> roundParams{
        CeilingParams(ngraph::PartialShape{3},
                     IN_ET,
                     std::vector<T>{0, 1, 0x4000000000000001},
                     std::vector<T>{0, 1, 0x4000000000000001})
    };
    return roundParams;
}

template <element::Type_t IN_ET>
std::vector<CeilingParams> generateParamsForCeilingInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<CeilingParams> roundParams{
        CeilingParams(ngraph::PartialShape{4},
                     IN_ET,
                     std::vector<T>{2, 136314888, 0x40000010, 0x40000001},
                     std::vector<T>{2, 136314888, 0x40000010, 0x40000001})
    };
    return roundParams;
}

std::vector<CeilingParams> generateCombinedParamsForCeiling() {
    const std::vector<std::vector<CeilingParams>> roundTypeParams{
        generateParamsForCeilingFloat<element::Type_t::f32>(),
        generateParamsForCeilingFloat<element::Type_t::f16>(),
        generateParamsForCeilingInt64<element::Type_t::i64>(),
        generateParamsForCeilingInt32<element::Type_t::i32>(),
        generateParamsForCeilingInt64<element::Type_t::u64>(),
        generateParamsForCeilingInt32<element::Type_t::u32>()
    };

    std::vector<CeilingParams> combinedParams;

    for (const auto& params : roundTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Ceiling_With_Hardcoded_Refs, 
    ReferenceCeilingLayerTest,
    ::testing::ValuesIn(generateCombinedParamsForCeiling()),
    ReferenceCeilingLayerTest::getTestCaseName);

}  // namespace
