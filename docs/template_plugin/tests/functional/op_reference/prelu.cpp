// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <algorithm>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct PreluParams {
    template <class IT>
    PreluParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues,
                const Shape& slopeShape, const std::vector<IT>& negativeSlopeValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)),
          negativeSlopeShape(slopeShape),
          negativeSlope(CreateBlob(iType, negativeSlopeValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
    Shape negativeSlopeShape;
    Blob::Ptr negativeSlope;
};

class ReferencePreluLayerTest : public testing::TestWithParam<PreluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.negativeSlope};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PreluParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "slopeShape=" << param.negativeSlopeShape;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PreluParams& params) {
        const auto in = std::make_shared<op::Parameter>(params.inType, params.pshape);
        //const auto SLOPE = op::Constant::create(input_type, negativeSlopeShape, negativeSlope->get());
        const auto SLOPE = std::make_shared<op::Parameter>(params.inType, params.negativeSlopeShape);
        const auto Prelu = std::make_shared<op::v0::PRelu>(in, SLOPE);
        return std::make_shared<Function>(NodeVector {Prelu}, ParameterVector {in, SLOPE});
    }
};

TEST_P(ReferencePreluLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PreluParams> generatePreluFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PreluParams> preluParams {
        PreluParams(ngraph::PartialShape {6},
                    IN_ET,
                    std::vector<T>{1, 2, -3, -4, 5, 6},
                    std::vector<T>{1, 2, -6, -8, 5, 6},
                    {1},
                    {2})
    };
    return preluParams;
}

std::vector<PreluParams> generatePreluCombinedParams() {
    const std::vector<std::vector<PreluParams>> preluTypeParams {
        generatePreluFloatParams<element::Type_t::f32>(),
        generatePreluFloatParams<element::Type_t::f16>()
        };
    std::vector<PreluParams> combinedParams;

    for (const auto& params : preluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Prelu_With_Hardcoded_Refs, ReferencePreluLayerTest,
    testing::ValuesIn(generatePreluCombinedParams()), ReferencePreluLayerTest::getTestCaseName);

} // namespace