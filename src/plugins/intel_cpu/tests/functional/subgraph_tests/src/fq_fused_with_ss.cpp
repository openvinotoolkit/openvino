// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using FQScaleshiftWithConstantShiftTestParams = Precision;

class FQScaleshiftWithConstantShiftTest : public testing::WithParamInterface<FQScaleshiftWithConstantShiftTestParams>,
                                       public CPUTestsBase,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FQScaleshiftWithConstantShiftTestParams> obj) {
        Precision netPrecision;
        netPrecision = obj.param;
        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        Precision netPrecision;
        netPrecision = this->GetParam();
        const auto ngPrec = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::Shape mmShape{25, 14, 14, 768};
        SizeVector mmShape2{768, 2304};
        SizeVector sumShape{1, 1, 1, 2304};

        // avoid eliminations
        std::vector<int> mmInData(768 * 2304);
        std::fill(mmInData.begin(), mmInData.end(), 2);
        mmInData[0] = 1;
        std::vector<int> sumConstData(1 * 1 * 1 * 2304);
        std::iota(sumConstData.begin(), sumConstData.end(), 0);

        auto constShift = ngraph::opset5::Constant::create(ngraph::element::f32, sumShape, sumConstData);
        auto mmConst = ngraph::opset5::Constant::create(ngraph::element::f32, mmShape2, mmInData);
        ov::ParameterVector mmParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, mmShape)};
        const auto mmOutputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(mmParams));

        const auto mm = builder::makeMatMul(mmOutputNodes[0], mmConst, false, false);
        auto sum = ngraph::builder::makeEltwise(constShift, mm, ngraph::helpers::EltwiseTypes::ADD);
        auto fq = ngraph::builder::makeFakeQuantize(sum, ngraph::element::f32, 256, {}, {-8.0f}, {7.0f}, {-8.0f}, {7.0f});

        ngraph::ParameterVector inputParams = {mmParams[0]};
        function = makeNgraphFunction(ngPrec, inputParams, fq, "FQScaleshiftWithConstantShift");
    }
};

/* Network with SS subgraph and FQ node. Shift in SS is constant-folded.
 * Test that FQ-SS fusing works correctly while comparing SS and FQ channel dims.
     Input         Const
          \       /
           \     /
            \   /
            MatMul      Const
               \         /
                \       /
                 \     /
                   Add
                    |
                    |
                   FQ
                    |
                    |
                 Output
*/

TEST_P(FQScaleshiftWithConstantShiftTest, CompareWithRefs) {
    Run();
}

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Check, FQScaleshiftWithConstantShiftTest,
                         ::testing::Values(Precision::FP32),
                         FQScaleshiftWithConstantShiftTest::getTestCaseName);
} // namespace
} // namespace SubgraphTestsDefinitions
