// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_norm_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class NormLayerBuilderTest : public BuilderTestCommon {};

TEST_F(NormLayerBuilderTest, getExistsLayerFromNetworkBuilderWithAcrossMapsEqualTrue) {
    Builder::Network net("Test");
    auto layer = Builder::NormLayer("NormLayer").setAlpha(9.999999747378752e-05f).setBeta(0.75f).setSize(5).setAcrossMaps(true).setPort(Port({10, 10, 100, 100}));
    size_t id = net.addLayer(layer);
    Builder::NormLayer layerFromNetwork(net.getLayer(id));
    ASSERT_EQ(layer.getAlpha(), layerFromNetwork.getAlpha());
    ASSERT_EQ(layer.getBeta(), layerFromNetwork.getBeta());
    ASSERT_EQ(layer.getAcrossMaps(), layerFromNetwork.getAcrossMaps());
}

TEST_F(NormLayerBuilderTest, getExistsLayerFromNetworkBuilderWithAcrossMapsEqualFalse) {
    Builder::Network net("Test");
    auto layer = Builder::NormLayer("NormLayer").setAlpha(9.999999747378752e-05f).setBeta(0.75f).setSize(5).setAcrossMaps(false).setPort(Port({10, 10, 100, 100}));
    size_t id = net.addLayer(layer);
    Builder::NormLayer layerFromNetwork(net.getLayer(id));
    ASSERT_EQ(layer.getAlpha(), layerFromNetwork.getAlpha());
    ASSERT_EQ(layer.getBeta(), layerFromNetwork.getBeta());
    ASSERT_EQ(layer.getAcrossMaps(), layerFromNetwork.getAcrossMaps());
}

TEST_F(NormLayerBuilderTest, cannotCreateNormLayerWithWrongAlpha) {
    Builder::Network net("Test");
    auto layer = Builder::NormLayer("NormLayer").setAlpha(0).setBeta(0.75f).setSize(5).setAcrossMaps(true).setPort(Port({10, 10, 100, 100}));
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormLayerBuilderTest, cannotCreateNormLayerWithWrongBeta) {
    Builder::Network net("Test");
    auto layer = Builder::NormLayer("NormLayer").setAlpha(1).setBeta(0).setSize(5).setAcrossMaps(true).setPort(Port({10, 10, 100, 100}));
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormLayerBuilderTest, cannotCreateNormLayerWithWrongSize) {
    Builder::Network net("Test");
    auto layer = Builder::NormLayer("NormLayer").setAlpha(1).setBeta(1).setSize(0).setAcrossMaps(true).setPort(Port({10, 10, 100, 100}));
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NormLayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeNormLayerPtr = std::make_shared<Builder::Layer>("Norm", "Norm layer");
    fakeNormLayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeNormLayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::NormLayer normLayer(fakeNormLayerPtr);
    normLayer.setAlpha(1).setBeta(0).setSize(5).setAcrossMaps(true);
    ASSERT_THROW(net.addLayer(normLayer), InferenceEngine::details::InferenceEngineException);
}

