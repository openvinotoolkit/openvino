// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_tanh_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class TanHLayerBuilderTest : public BuilderTestCommon {};

TEST_F(TanHLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::TanHLayer tanhLayer("TanH_layer");
    size_t ind = net.addLayer(tanhLayer);
    Builder::TanHLayer layerFromNet(net.getLayer(ind));
}

TEST_F(TanHLayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeTanHLayerPtr = std::make_shared<Builder::Layer>("TanH", "TanH layer");
    fakeTanHLayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeTanHLayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::TanHLayer tanhLayer(fakeTanHLayerPtr);
    ASSERT_THROW(net.addLayer(tanhLayer), InferenceEngine::details::InferenceEngineException);
}