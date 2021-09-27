// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/stress_tests.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace LayerTestsDefinitions {

std::string MultipleAllocations::getTestCaseName(const testing::TestParamInfo<MultipleAllocationsParams>& obj) {
    LayerTestsUtils::TargetDevice targetDevice;
    unsigned int allocationsCount;

    std::tie(targetDevice, allocationsCount) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << targetDevice << "_";
    result << "allocationsCount=" << allocationsCount;
    return result.str();
}

void MultipleAllocations::SetUp() {
    std::tie(targetDevice, m_allocationsCount) = this->GetParam();
    function = ngraph::builder::subgraph::makeSplitConvConcat();
}

TEST_P(MultipleAllocations, InferWorksCorrectAfterAllocations) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::CNNNetwork cnnNet(function);
    auto ie = PluginCache::get().ie();

    std::cout << "Load the network " << m_allocationsCount << " times..." << std::flush;
    for (unsigned int i = 0; i < m_allocationsCount; ++i) {
        ie->LoadNetwork(cnnNet, targetDevice, configuration);
    }

    std::cout << "\nCheck inference.\n";

    // Experiments demonstrated that 10 cycles are enough to reproduce the issue
    int infersCount = 10;
    for (int j = 0; j < infersCount; ++j) {
        LoadNetwork();

        std::cout << "Infer(): " << j << std::flush;
        if (j == 0) {
            GenerateInputs();
        }
        Infer();
        Validate();
    }
};

} // namespace LayerTestsDefinitions
