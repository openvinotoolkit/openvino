// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_mvn_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class MVNLayerBuilderTest : public BuilderTestCommon {};

TEST_F(MVNLayerBuilderTest, getExistsLayerFromNetworkBuilder1) {
    Builder::Network net("network");
    Builder::MVNLayer mvnLayer("MVN_layer");
    mvnLayer.setEpsilon(99.9).setAcrossChannels(true).setNormalize(true);
    size_t ind = net.addLayer(mvnLayer);
    Builder::MVNLayer layerFromNet(net.getLayer(ind));
}

TEST_F(MVNLayerBuilderTest, getExistsLayerFromNetworkBuilder2) {
    Builder::Network net("network");
    Builder::MVNLayer mvnLayer("MVN_layer");
    mvnLayer.setEpsilon(99.9).setAcrossChannels(true).setNormalize(false);
    size_t ind = net.addLayer(mvnLayer);
    Builder::MVNLayer layerFromNet(net.getLayer(ind));
}

TEST_F(MVNLayerBuilderTest, getExistsLayerFromNetworkBuilder3) {
    Builder::Network net("network");
    Builder::MVNLayer mvnLayer("MVN_layer");
    mvnLayer.setEpsilon(99.9).setAcrossChannels(false).setNormalize(true);
    size_t ind = net.addLayer(mvnLayer);
    Builder::MVNLayer layerFromNet(net.getLayer(ind));
}

TEST_F(MVNLayerBuilderTest, getExistsLayerFromNetworkBuilder4) {
    Builder::Network net("network");
    Builder::MVNLayer mvnLayer("MVN_layer");
    mvnLayer.setEpsilon(99.9).setAcrossChannels(false).setNormalize(false);
    size_t ind = net.addLayer(mvnLayer);
    Builder::MVNLayer layerFromNet(net.getLayer(ind));
}

TEST_F(MVNLayerBuilderTest, cannotCreateLayerWithWrongEpsion) {
    Builder::Network net("network");
    Builder::MVNLayer mvnLayer("MVN_layer");
    mvnLayer.setEpsilon(-100).setAcrossChannels(true).setNormalize(true);  // here
    ASSERT_THROW(net.addLayer(mvnLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(MVNLayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeMVNLayerPtr = std::make_shared<Builder::Layer>("MVN", "MVN layer");
    fakeMVNLayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeMVNLayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::MVNLayer mvnLayer(fakeMVNLayerPtr);
    mvnLayer.setEpsilon(100).setAcrossChannels(true).setNormalize(true);
    ASSERT_THROW(net.addLayer(mvnLayer), InferenceEngine::details::InferenceEngineException);
}