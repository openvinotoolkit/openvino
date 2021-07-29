// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ngraph_functions/subgraph_builders.hpp>

#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

using namespace InferenceEngine;

PRETTY_PARAM(num_graphs, int)
typedef myriadLayerTestBaseWithParam<num_graphs> myriadMultipleGraphsTests_nightly;

// Test ability to load many graphs to device
TEST_P(myriadMultipleGraphsTests_nightly, LoadGraphsOnDevice) {
    ASSERT_NO_THROW(_cnnNetwork = InferenceEngine::CNNNetwork(ngraph::builder::subgraph::makeSplitConvConcat()));
    const int num_graphs = GetParam();
    std::vector<InferenceEngine::ExecutableNetwork> exeNetworks(num_graphs);
    for (int i = 0; i < num_graphs; ++i) {
        ASSERT_NO_THROW(exeNetworks[i] = _vpuPluginPtr->LoadNetwork(_cnnNetwork));
    }
}

INSTANTIATE_TEST_SUITE_P(numerOfGraphs, myriadMultipleGraphsTests_nightly,
    ::testing::Values(2, 4, 10)
);
