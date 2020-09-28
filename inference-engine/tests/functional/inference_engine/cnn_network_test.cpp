// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cpp/ie_cnn_network.h"

using namespace InferenceEngine;

using CNNNetworkTests = ::testing::Test;

TEST_F(CNNNetworkTests, throwsOnInitWithNull) {
    std::shared_ptr<ICNNNetwork> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnInitWithNullNgraph) {
    std::shared_ptr<ngraph::Function> nlptr = nullptr;
    ASSERT_THROW(CNNNetwork network(nlptr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetOutputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getOutputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetInputsInfo) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputsInfo(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedLayerCount) {
    CNNNetwork network;
    ASSERT_THROW(network.layerCount(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetName) {
    CNNNetwork network;
    ASSERT_THROW(network.getName(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedCastToICNNNetwork) {
    CNNNetwork network;
    ASSERT_THROW(auto & net = static_cast<ICNNNetwork&>(network), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedCastToICNNNetwork) {
    const CNNNetwork network;
    ASSERT_THROW(const auto & net = static_cast<const ICNNNetwork&>(network), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnUninitializedGetFunction) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedGetFunction) {
    const CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedBegin) {
    CNNNetwork network;
    ASSERT_THROW(network.getFunction(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CNNNetworkTests, throwsOnConstUninitializedGetInputShapes) {
    CNNNetwork network;
    ASSERT_THROW(network.getInputShapes(), InferenceEngine::details::InferenceEngineException);
}
