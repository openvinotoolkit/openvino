// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_clamp_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ClampLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ClampLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::ClampLayer clampLayer("clampLayer");
    clampLayer.setMinValue(0.1).setMaxValue(0.2);
    size_t ind = net.addLayer(clampLayer);
    Builder::ClampLayer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(layerFromNet.getMinValue(), clampLayer.getMinValue());
    ASSERT_EQ(layerFromNet.getMaxValue(), clampLayer.getMaxValue());
}

TEST_F(ClampLayerBuilderTest, cannotCreateLayerWithWrongMinValue) {
    Builder::Network net("network");
    Builder::ClampLayer clampLayer("clampLayer");
    clampLayer.setMinValue(0).setMaxValue(0.2);
    ASSERT_NO_THROW(net.addLayer(clampLayer));
}

TEST_F(ClampLayerBuilderTest, cannotCreateLayerWithWrongMaxValue) {
    Builder::Network net("network");
    Builder::ClampLayer clampLayer("clampLayer");
    clampLayer.setMinValue(10).setMaxValue(-0.2);
    ASSERT_THROW(net.addLayer(clampLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ClampLayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeClampLayerPtr = std::make_shared<Builder::Layer>("Clamp", "Clamp layer");
    fakeClampLayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeClampLayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::ClampLayer clampLayer(fakeClampLayerPtr);
    clampLayer.setMinValue(0.0f).setMaxValue(1.0f);
    ASSERT_THROW(net.addLayer(clampLayer), InferenceEngine::details::InferenceEngineException);
}