// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ngraph;
using ngraph::helpers::EltwiseTypes;

namespace SubgraphTestsDefinitions {

class TileWithTwoOutputEdges : public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        auto ngPrc = element::f32;
        auto inputParams = builder::makeParams(ngPrc, {{1, 3, 12, 9}});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));
        outPrc.front() = InferenceEngine::Precision::FP32;
        outPrc.push_back(InferenceEngine::Precision::FP32);
        auto tile = ngraph::builder::makeTile(paramOuts[0], std::vector<int64_t>{1, 2, 1, 1});

        const auto const1 = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, 6, 1, 1}, std::vector<float>{}, true);
        const auto const2 = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, 6, 1, 1}, std::vector<float>{}, true);

        const auto add1 = ngraph::builder::makeEltwise(tile->output(0), const1, ngraph::helpers::EltwiseTypes::ADD);
        const auto add2 = ngraph::builder::makeEltwise(tile->output(0), const2, ngraph::helpers::EltwiseTypes::ADD);

        NodeVector results{add1, add2};
        function = std::make_shared<ngraph::Function>(results, inputParams, "TileWithTwoOutputEdges");
    }
};

TEST_F(TileWithTwoOutputEdges, smoke_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

} // namespace SubgraphTestsDefinitions