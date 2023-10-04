// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using MatmulStridedInputsOutputsTestParams = Precision;

class MatmulStridedInputsOutputsTest : public testing::WithParamInterface<MatmulStridedInputsOutputsTestParams>,
                                       public CPUTestsBase,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulStridedInputsOutputsTestParams> obj) {
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

        SizeVector splitShape{1, 2, 1, 16};
        ov::ParameterVector splitInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(splitShape))};
        const auto splitOutputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(splitInputParams));
        const auto split = builder::makeSplit(splitOutputNodes[0], ngPrec, 2 /* splits */, 1 /* 2nd axis */);

        std::vector<ov::Shape> concatShapes{{1, 1, 8, 8}, {1, 1, 8, 8}};
        ov::ParameterVector concatInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[0]),
                                               std::make_shared<ov::op::v0::Parameter>(ngPrec, concatShapes[1])};
        const auto concatOutputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(concatInputParams));
        const auto concat = builder::makeConcat(concatOutputNodes, 2);

        const auto matMul1 = builder::makeMatMul(split->output(0), concat, false, false);

        SizeVector matmulShape{1, 1, 16, 8};
        ov::ParameterVector matmulInputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(matmulShape))};
        const auto matmulOutputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(matmulInputParams));

        const auto matMul2 = builder::makeMatMul(split->output(1), matmulOutputNodes[0], false, false);

        const auto concatMatMuls = builder::makeConcat({matMul1, matMul2}, 2 /* 3rd axis */);

        ngraph::ParameterVector inputParams = {splitInputParams[0], concatInputParams[0], concatInputParams[1], matmulInputParams[0]};
        function = makeNgraphFunction(ngPrec, inputParams, concatMatMuls, "MatmulStridedInputsOutputs");
    }
};

/* Network with two MatMul nodes and multiple inplace nodes
 * Test that MatMul node works correctly with strided inputs / outputs

   Input    Input Input
     \       /      |
      \     /       |
       \   /        |
        \ /         |
       Concat     Split      Input
          \       /   \       /
           \     /     \     /
            \   /       \   /
             \ /         \ /
            MatMul     MatMul
               \         /
                \       /
                 \     /
                  \   /
                 Concat
                    |
                    |
                 Output
*/

TEST_P(MatmulStridedInputsOutputsTest, CompareWithRefs) {
    Run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Check, MatmulStridedInputsOutputsTest,
                         ::testing::Values(Precision::FP32,
                                           Precision::BF16),
                         MatmulStridedInputsOutputsTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
