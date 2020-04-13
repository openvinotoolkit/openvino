// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_relu6_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ReLU6LayerBuilderTest : public BuilderTestCommon {};

TEST_F(ReLU6LayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::ReLU6Layer relu6Layer("relu6layer");
    relu6Layer.setN(100);
    size_t ind = net.addLayer(relu6Layer);
    Builder::ReLU6Layer layerFromNet(net.getLayer(ind));
    ASSERT_EQ(relu6Layer.getN(), layerFromNet.getN());
}

TEST_F(ReLU6LayerBuilderTest, cannotCreateLayerWithWrongShapes) {
    Builder::Network net("network");
    Builder::Layer::Ptr fakeReLU6LayerPtr = std::make_shared<Builder::Layer>("ReLU6", "ReLU6 layer");
    fakeReLU6LayerPtr->getInputPorts().push_back(Port({1, 1, 1, 1}));
    fakeReLU6LayerPtr->getOutputPorts().push_back(Port({1, 1, 1, 2}));
    Builder::ReLU6Layer reLU6Layer(fakeReLU6LayerPtr);
    reLU6Layer.setN(10);
    ASSERT_THROW(net.addLayer(reLU6Layer), InferenceEngine::details::InferenceEngineException);
}