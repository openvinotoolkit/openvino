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
struct HardSigmoidParams {
    template <class IT>
    HardSigmoidParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues,
                const double alpha, const double beta)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)),
          alpha(alpha),
          beta(beta) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
    double alpha;
    double beta;
};

class ReferenceHardSigmoidLayerTest : public testing::TestWithParam<HardSigmoidParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.alpha, params.beta);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<HardSigmoidParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "alpha=" << param.alpha << "_";
        result << "beta=" << param.beta;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type, const double alphaData, const double betaData) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto alpha = op::Constant::create(input_type, Shape{}, {alphaData});
        const auto beta = op::Constant::create(input_type, Shape{}, {betaData});
        const auto HardSigmoid = std::make_shared<op::v0::HardSigmoid>(in, alpha, beta);
        return std::make_shared<Function>(NodeVector {HardSigmoid}, ParameterVector {in});
    }
};

TEST_P(ReferenceHardSigmoidLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<HardSigmoidParams> generateHardSigmoidFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<HardSigmoidParams> hardSigmoidParams {
        HardSigmoidParams(ngraph::PartialShape {3},
                    IN_ET,
                    std::vector<T>{-1.0f, 0.0f, 1.0f},
                    std::vector<T>{0.1f, 0.6f, 1.f},
                    0.5,
                    0.6),
        HardSigmoidParams(ngraph::PartialShape {2, 5},
                    IN_ET,
                    std::vector<T>{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 0.5f, -0.2f, 6.0f, 8.0f, 0.1f},
                    std::vector<T>{0.0f, 0.3f, 0.5f, 0.7f, 1.0f, 0.6f, 0.46f, 1.0f, 1.0f, 0.52f},
                    0.2,
                    0.5)
    };
    return hardSigmoidParams;
}

std::vector<HardSigmoidParams> generateHardSigmoidCombinedParams() {
    const std::vector<std::vector<HardSigmoidParams>> hardSigmoidTypeParams {
        generateHardSigmoidFloatParams<element::Type_t::f32>(),
        generateHardSigmoidFloatParams<element::Type_t::f16>()
        };
    std::vector<HardSigmoidParams> combinedParams;

    for (const auto& params : hardSigmoidTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_HardSigmoid_With_Hardcoded_Refs, ReferenceHardSigmoidLayerTest,
    testing::ValuesIn(generateHardSigmoidCombinedParams()), ReferenceHardSigmoidLayerTest::getTestCaseName);

} // namespace