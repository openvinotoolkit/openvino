// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ReshapeFCTestParams = std::tuple<std::pair<SizeVector, SizeVector>, // IS fully connected
                                       bool,                              // transpose B
                                       fusingSpecificParams>;

class ReshapeFCTest : public testing::WithParamInterface<ReshapeFCTestParams>, public CpuTestWithFusing,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReshapeFCTestParams> obj) {
        std::pair<SizeVector, SizeVector> isFc;
        bool transpB;
        fusingSpecificParams fusingParams;
        std::tie(isFc, transpB, fusingParams) = obj.param;
        SizeVector isA = isFc.first; SizeVector isB = isFc.second;

        std::ostringstream result;
        result << "IS_reshape=" << CommonTestUtils::vec2str(isA) << "_";
        result << "IS_fc_B=" << CommonTestUtils::vec2str(isB) << "_";
        result << "Transp_B=" << transpB;
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        std::pair<SizeVector, SizeVector> isFc;
        bool transpB;
        fusingSpecificParams fusingParams;
        std::tie(isFc, transpB, fusingParams) = this->GetParam();
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;
        SizeVector isReshape = isFc.first; SizeVector isB = isFc.second;
        SizeVector isA(2);
        isA[0] = isReshape[0];
        isA[1] = std::accumulate(isReshape.begin() + 1, isReshape.end(), size_t{1}, std::multiplies<size_t>());
        if (transpB) {
            std::swap(*(isB.end() - 1), *(isB.end() - 2));
        }

        auto inputParams = builder::makeParams(element::f32, {isReshape});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        auto constNode = builder::makeConstant(element::i64, {isA.size()}, isA);
        auto reshape = std::make_shared<opset1::Reshape>(paramOuts[0], constNode, true);

        auto matrixB = builder::makeConstant<float>(element::f32, isB, {}, true);
        auto matMul = builder::makeMatMul(reshape, matrixB, false, transpB);

        function = makeNgraphFunction(element::f32, inputParams, matMul, "ReshapeFC");
    }
};

TEST_P(ReshapeFCTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckNodeOfTypeCount(executableNetwork, "Reshape", 0);
    CheckFusingResults(executableNetwork, "FullyConnected");
}

namespace {

const std::vector<bool> transpose = {
    true, false
};

const std::vector<std::pair<SizeVector, SizeVector>> isFC = {
    {{71, 128, 1, 1}, {128, 20}},
    {{1, 24, 2, 7}, {336, 16}}
};

std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingAddPerChannel
};

const auto reshapeFCParams = ::testing::Combine(::testing::ValuesIn(isFC),
                                                ::testing::ValuesIn(transpose),
                                                ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_CASE_P(smoke_Check, ReshapeFCTest, reshapeFCParams, ReshapeFCTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
