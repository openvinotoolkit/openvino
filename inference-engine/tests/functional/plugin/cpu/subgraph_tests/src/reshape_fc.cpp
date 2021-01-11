// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ReshapeFCTestParams = std::tuple<SizeVector,                        // IS reshape
                                       std::pair<SizeVector, SizeVector>, // IS fully connected weight
                                       bool>;                             // transpose B

class ReshapeFCTest : public testing::WithParamInterface<ReshapeFCTestParams>, public CPUTestsBase,
                      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReshapeFCTestParams> obj) {
        SizeVector isReshape;
        std::pair<SizeVector, SizeVector> isFc;
        bool transpB;
        std::tie(isReshape, isFc, transpB) = obj.param;
        SizeVector isA = isFc.first; SizeVector isB = isFc.second;

        std::ostringstream result;
        result << "IS_reshape=" << CommonTestUtils::vec2str(isReshape) << "_";
        result << "IS_fc_A=" << CommonTestUtils::vec2str(isA) << "_";
        result << "IS_fc_B=" << CommonTestUtils::vec2str(isB) << "_";
        result << "Transp_B=" << transpB;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        SizeVector isReshape;
        std::pair<SizeVector, SizeVector> isFc;
        bool transpB;
        std::tie(isReshape, isFc, transpB) = this->GetParam();
        SizeVector isA = isFc.first; SizeVector isB = isFc.second;

        if (transpB) {
            std::swap(*(isB.end() - 1), *(isB.end() - 2));
        }

        auto inputParams = builder::makeParams(element::f32, {isReshape});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        auto constNode = builder::makeConstant(element::i64, {isA.size()}, isA);
        auto reshape = std::make_shared<opset1::Reshape>(paramOuts[0], constNode, true);

        auto matrixB = builder::makeConstant<float>(element::f32, isB, {}, true);
        auto matMul = builder::makeMatMul(reshape, matrixB, false, transpB);

        ResultVector results{std::make_shared<ngraph::opset1::Result>(matMul)};
        function = std::make_shared<Function>(results, inputParams, "ReshapeFC");
    }
};

TEST_P(ReshapeFCTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckNodeOfTypeCount(executableNetwork, "Reshape", 0);
}

namespace {

const std::vector<bool> transpose = {
    true, false
};

const std::vector<SizeVector> isRehsape = {
    {71, 128, 1, 1}
};

const std::vector<std::pair<SizeVector, SizeVector>> isFC = {
    {{71, 128}, {128, 20}}
};

const auto reshapeFCParams = ::testing::Combine(::testing::ValuesIn(isRehsape),
                                                ::testing::ValuesIn(isFC),
                                                ::testing::ValuesIn(transpose));

INSTANTIATE_TEST_CASE_P(smoke_Check, ReshapeFCTest, reshapeFCParams, ReshapeFCTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
