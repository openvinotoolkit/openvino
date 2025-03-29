// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cassert>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using AlignMatMulInputRanksTestParams = std::tuple<std::pair<ov::Shape, ov::Shape>,  // IS fully connected
                                                   fusingSpecificParams>;

class AlignMatMulInputRanksTest : public testing::WithParamInterface<AlignMatMulInputRanksTestParams>,
                                  public CpuTestWithFusing,
                                  virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AlignMatMulInputRanksTestParams> obj) {
        std::pair<ov::Shape, ov::Shape> supportedInputShapes;
        fusingSpecificParams fusingParams;
        std::tie(supportedInputShapes, fusingParams) = obj.param;
        ov::Shape inputShapeA = supportedInputShapes.first;
        ov::Shape inputShapeB = supportedInputShapes.second;

        std::ostringstream result;
        result << "IS_A=" << inputShapeA << "_";
        result << "IS_B=" << inputShapeB << "_";
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::pair<ov::Shape, ov::Shape> inShapes;
        fusingSpecificParams fusingParams;
        std::tie(inShapes, fusingParams) = this->GetParam();

        if (inShapes.first.size() != inShapes.second.size())
            expectedNumOfReshapes++;  // one input will be unsqueezed
        if (inShapes.first.size() == 1 || inShapes.second.size() == 1)
            expectedNumOfReshapes++;  // output will be squeezed
        if (inShapes.first.size() == 1 && inShapes.second.size() == 1)
            expectedNumOfReshapes += 2;  // both inputs unsqueezed and output squeezed

        if (inShapes.first.size() != 1 && inShapes.second.size() != 1)  // no fusing through Reshape after output
            std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        const auto ngPrec = element::f32;
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ngPrec, inShapes.first),
                                        std::make_shared<ov::op::v0::Parameter>(ngPrec, inShapes.second)};
        const auto matMul = std::make_shared<ov::op::v0::MatMul>(inputParams[0], inputParams[1], false, false);

        function = makeNgraphFunction(ngPrec, inputParams, matMul, "AlignMatMulInputRanks");
    }

    int expectedNumOfReshapes = 0;
};

TEST_P(AlignMatMulInputRanksTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel,
                               "Reshape",
                               expectedNumOfReshapes);  // Squeeze / Unsqueeze turns into Reshape
}

namespace {

const std::vector<std::pair<ov::Shape, ov::Shape>> supportedInputShapes = {
    {{4, 10, 5}, {1, 5, 10}},       // nothing to be done
    {{3}, {3}},                     // 3x1 * 1x3 -> 1
    {{18}, {1, 5, 18, 20}},         // 1x1x1x18 * 1x5x18x20 -> 1x5x20
    {{2, 3, 4, 4, 4, 10, 5}, {5}},  // 2x3x4x4x4x10x5 * 1x1x1x1x1x5x1 -> 1x1x1x1x1x5
    {{1, 18}, {1, 5, 18, 20}},
    {{1, 70, 18}, {1, 5, 18, 20}},
    {{7, 1, 10, 3, 2, 7}, {1, 7, 5}},
    {{2, 3, 4, 4, 4, 10, 5}, {5, 20}},
};

// verify fusing just in case
std::vector<fusingSpecificParams> fusingParamsSet{
    emptyFusingSpec,
    fusingElu,
};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         AlignMatMulInputRanksTest,
                         ::testing::Combine(::testing::ValuesIn(supportedInputShapes),
                                            ::testing::ValuesIn(fusingParamsSet)),
                         AlignMatMulInputRanksTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
