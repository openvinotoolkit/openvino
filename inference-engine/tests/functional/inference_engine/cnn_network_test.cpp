// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cpp/ie_cnn_network.h"

using namespace InferenceEngine;

using CNNNetworkTests = ::testing::Test;

TEST_F(CNNNetworkTests, smoke_throwsOnInitWithNull) {
    std::shared_ptr<ICNNNetwork> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnInitWithNullNgraph) {
    std::shared_ptr<ngraph::Function> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedGetOutputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedGetInputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedLayerCount) {
    CNNNetwork network;
    ASSERT_THROW(network.layerCount(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedGetName) {
    CNNNetwork network;
    ASSERT_THROW(network.getName(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedCastToICNNNetwork) {
    CNNNetwork network;
    ASSERT_THROW(auto & net = static_cast<ICNNNetwork&>(network), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnConstUninitializedCastToICNNNetwork) {
    const CNNNetwork network;
    ASSERT_THROW(const auto & net = static_cast<const ICNNNetwork&>(network), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnUninitializedGetFunction) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnConstUninitializedGetFunction) {
    const CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnConstUninitializedBegin) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, smoke_throwsOnConstUninitializedGetInputShapes) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputShapes(), InferenceEngine::details::InferenceEngineException);
}
