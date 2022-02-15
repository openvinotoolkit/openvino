// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ngraph;
using ngraph::helpers::EltwiseTypes;

namespace SubgraphTestsDefinitions {

class NotFusedConvSimpleOp : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        auto inputParams = builder::makeParams(element::f32, {{1, 3, 12, 9}, {1, 16, 12, 9}});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));
        outPrc.front() = InferenceEngine::Precision::FP32;
        outPrc.push_back(InferenceEngine::Precision::FP32);
        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {3, 3};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 16;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = builder::makeConvolution(paramOuts[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
        }
        const auto sharedNode = builder::makeConstant(element::f32, {1, 16, 1, 1}, std::vector<float>{}, true);
        const auto postOpCandidate = builder::makeEltwise(conv, sharedNode, EltwiseTypes::ADD);

        const auto secondConsumpt = builder::makeEltwise(paramOuts[1], sharedNode, EltwiseTypes::ADD);

        NodeVector results{postOpCandidate, secondConsumpt};
        function = std::make_shared<ngraph::Function>(results, inputParams, "NotFusedConvSimpleOp");
    }
};

TEST_F(NotFusedConvSimpleOp, smoke_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

} // namespace SubgraphTestsDefinitions
