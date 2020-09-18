// Copyright (C) 2018-2020 Intel Corporation
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
    StatusCode st;
    std::vector<InferenceEngine::IExecutableNetwork::Ptr> exeNetwork(num_graphs);
    std::map<std::string, std::string> networkConfig;
    for (int i = 0; i < num_graphs; ++i) {
        st = _vpuPluginPtr->LoadNetwork(exeNetwork[i], _cnnNetwork, networkConfig, &_resp);
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }
}

INSTANTIATE_TEST_CASE_P(numerOfGraphs, myriadMultipleGraphsTests_nightly,
    ::testing::Values(2, 4, 10)
);
