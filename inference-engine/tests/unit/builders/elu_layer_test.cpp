// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_elu_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ELULayerBuilderTest : public BuilderTestCommon {};

TEST_F(ELULayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::ELULayer eluLayer("ELU_layer");
    eluLayer.setAlpha(100);
    size_t ind = net.addLayer(eluLayer);
    Builder::ELULayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(eluLayer.getAlpha(), layerFromNet.getAlpha());
}

TEST_F(ELULayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeELULayerPtr = std::make_shared<Builder::Layer>("ELU", "ELU layer");
    fakeELULayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeELULayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::ELULayer eluLayer(fakeELULayerPtr);
    eluLayer.setAlpha(100);
    ASSERT_THROW(net.addLayer(eluLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ELULayerBuilderTest, cannotCreateLayerWithWrongAlpha) {
    Builder::Network net("network");
    Builder::ELULayer eluLayer("ELU_layer");
    eluLayer.setAlpha(-100);
    ASSERT_THROW(net.addLayer(eluLayer), InferenceEngine::details::InferenceEngineException);
}