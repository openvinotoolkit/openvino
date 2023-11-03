// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/common_utils.hpp"

#include <algorithm>
#include <cassert>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using AlignMatMulInputRanksTestParams = std::tuple<std::pair<SizeVector, SizeVector>, // IS fully connected
                                       fusingSpecificParams>;

class AlignMatMulInputRanksTest : public testing::WithParamInterface<AlignMatMulInputRanksTestParams>, public CpuTestWithFusing,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AlignMatMulInputRanksTestParams> obj) {
        std::pair<SizeVector, SizeVector> supportedInputShapes;
        fusingSpecificParams fusingParams;
        std::tie(supportedInputShapes, fusingParams) = obj.param;
        SizeVector inputShapeA = supportedInputShapes.first; SizeVector inputShapeB = supportedInputShapes.second;

        std::ostringstream result;
        result << "IS_A=" << ov::test::utils::vec2str(inputShapeA) << "_";
        result << "IS_B=" << ov::test::utils::vec2str(inputShapeB) << "_";
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::pair<SizeVector, SizeVector> inShapes;
        fusingSpecificParams fusingParams;
        std::tie(inShapes, fusingParams) = this->GetParam();

        if (inShapes.first.size() != inShapes.second.size())
            expectedNumOfReshapes++;  // one input will be unsqueezed
        if (inShapes.first.size() == 1 || inShapes.second.size() == 1)
            expectedNumOfReshapes++;  // output will be squeezed
        if (inShapes.first.size() == 1 && inShapes.second.size() == 1)
            expectedNumOfReshapes+=2; // both inputs unsqueezed and output squeezed

        if (inShapes.first.size() != 1 && inShapes.second.size() != 1) // no fusing through Reshape after output
            std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        const auto ngPrec = element::f32;
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(inShapes.first)),
                                        std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(inShapes.second))};
        const auto matMul = builder::makeMatMul(inputParams[0], inputParams[1], false, false);

        selectedType = makeSelectedTypeStr(with_cpu_x86_avx512_core() ? "brgemm_avx512" : "jit_gemm", ngPrec);

        function = makeNgraphFunction(ngPrec, inputParams, matMul, "AlignMatMulInputRanks");
    }

    int expectedNumOfReshapes = 0;
};

TEST_P(AlignMatMulInputRanksTest, CompareWithRefs) {
    Run();
    CheckNumberOfNodesWithType(executableNetwork, "Reshape", expectedNumOfReshapes); // Squeeze / Unsqueeze turns into Reshape
    CheckPluginRelatedResults(executableNetwork, "MatMul");
}

namespace {

const std::vector<std::pair<SizeVector, SizeVector>> supportedInputShapes = {
    {{4, 10, 5}, {1, 5, 10}},      // nothing to be done
    {{3}, {3}},                    // 3x1 * 1x3 -> 1
    {{18}, {1, 5, 18, 20}},        // 1x1x1x18 * 1x5x18x20 -> 1x5x20
    {{2, 3, 4, 4, 4, 10, 5}, {5}}, // 2x3x4x4x4x10x5 * 1x1x1x1x1x5x1 -> 1x1x1x1x1x5
    {{1, 18}, {1, 5, 18, 20}},
    {{1, 70, 18}, {1, 5, 18, 20}},
    {{7, 1, 10, 3, 2, 7}, {1, 7, 5}},
    {{2, 3, 4, 4, 4, 10, 5}, {5, 20}},
};

// verify fusing just in case
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingElu,
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, AlignMatMulInputRanksTest,
                         ::testing::Combine(::testing::ValuesIn(supportedInputShapes),
                                            ::testing::ValuesIn(fusingParamsSet)),
                         AlignMatMulInputRanksTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
