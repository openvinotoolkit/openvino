// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_normalize_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class NormalizeLayerBuilderTest : public BuilderTestCommon {};

TEST_F(NormalizeLayerBuilderTest, getExistsLayerFromNetworkBuilder1) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0.1).setChannelShared(true).setAcrossMaps(true);
    size_t ind = net.addLayer(normalizeLayer);
    Builder::NormalizeLayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(normalizeLayer.getEpsilon(), layerFromNet.getEpsilon());
}

TEST_F(NormalizeLayerBuilderTest, getExistsLayerFromNetworkBuilder2) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0.1).setChannelShared(true).setAcrossMaps(false);
    size_t ind = net.addLayer(normalizeLayer);
    Builder::NormalizeLayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(normalizeLayer.getEpsilon(), layerFromNet.getEpsilon());
}

TEST_F(NormalizeLayerBuilderTest, getExistsLayerFromNetworkBuilder3) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0.1).setChannelShared(false).setAcrossMaps(true);
    size_t ind = net.addLayer(normalizeLayer);
    Builder::NormalizeLayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(normalizeLayer.getEpsilon(), layerFromNet.getEpsilon());
}

TEST_F(NormalizeLayerBuilderTest, getExistsLayerFromNetworkBuilder4) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0.1).setChannelShared(false).setAcrossMaps(false);
    size_t ind = net.addLayer(normalizeLayer);
    Builder::NormalizeLayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(normalizeLayer.getEpsilon(), layerFromNet.getEpsilon());
}

TEST_F(NormalizeLayerBuilderTest, cannotCreateLayerWithWrongEpsilon1) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0).setChannelShared(true).setAcrossMaps(true);
    ASSERT_THROW(net.addLayer(normalizeLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormalizeLayerBuilderTest, cannotCreateLayerWithWrongEpsilon2) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0).setChannelShared(true).setAcrossMaps(false);
    ASSERT_THROW(net.addLayer(normalizeLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormalizeLayerBuilderTest, cannotCreateLayerWithWrongEpsilon3) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0).setChannelShared(false).setAcrossMaps(true);
    ASSERT_THROW(net.addLayer(normalizeLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormalizeLayerBuilderTest, cannotCreateLayerWithWrongEpsilon4) {
    Builder::Network net("network");
    Builder::NormalizeLayer normalizeLayer("normalizeLayer");
    normalizeLayer.setEpsilon(0).setChannelShared(false).setAcrossMaps(false);
    ASSERT_THROW(net.addLayer(normalizeLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormalizeLayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeNormalizeLayerPtr = std::make_shared<Builder::Layer>("Normalize", "Normalize layer");
    fakeNormalizeLayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeNormalizeLayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::NormalizeLayer normalizeLayer(fakeNormalizeLayerPtr);
    normalizeLayer.setEpsilon(0.1).setChannelShared(true).setAcrossMaps(true);
    ASSERT_THROW(net.addLayer(normalizeLayer), InferenceEngine::details::InferenceEngineException);
}
