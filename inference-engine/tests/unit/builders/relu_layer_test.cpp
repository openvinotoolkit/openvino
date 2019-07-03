// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_relu_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ReLULayerBuilderTest : public BuilderTestCommon {};

TEST_F(ReLULayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::ReLULayer reluLayer("ReLU_layer");
    reluLayer.setNegativeSlope(100);
    size_t ind = net.addLayer(reluLayer);
    Builder::ReLULayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(reluLayer.getNegativeSlope(), layerFromNet.getNegativeSlope());
}

TEST_F(ReLULayerBuilderTest, cannotCreateLayerWithWrongNegativeSlope) {
    Builder::Network net("network");
    Builder::ReLULayer reluLayer("ReLU_layer");
    reluLayer.setNegativeSlope(-10);
    ASSERT_NO_THROW(net.addLayer(reluLayer));
}

TEST_F(ReLULayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeReLULayerPtr = std::make_shared<Builder::Layer>("ReLU", "ReLU layer");
    fakeReLULayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeReLULayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::ReLULayer reluLayer(fakeReLULayerPtr);
    reluLayer.setNegativeSlope(100);
    ASSERT_THROW(net.addLayer(reluLayer), InferenceEngine::details::InferenceEngineException);
}
