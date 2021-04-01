// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include "graph_tools_functional_tests.hpp"
#include <legacy/details/ie_cnn_network_tools.h>

using namespace testing;
using namespace InferenceEngine::details;
using namespace InferenceEngine;
using namespace std;

TEST_F(GraphToolsFncTest, smoke_canSortSplitConvConcat) {
    CNNNetwork network(ngraph::builder::subgraph::makeSplitConvConcat());
    checkSort(CNNNetSortTopologically(network));
}


TEST_F(GraphToolsFncTest, smoke_canSortTIwithLstm) {
    CNNNetwork network(ngraph::builder::subgraph::makeTIwithLSTMcell());
    checkSort(CNNNetSortTopologically(network));

    checkSort(CNNNetSortTopologically(network));
}