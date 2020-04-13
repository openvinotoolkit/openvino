// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_eltwise_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class EltwiseLayerBuilderTest : public BuilderTestCommon {};

TEST_F(EltwiseLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3, 4}), Port({1, 2, 3, 4})});
    layer.setOutputPort(Port({1, 2, 3, 4}));
    size_t ind = 0;
    ASSERT_NO_THROW(ind = net.addLayer(layer));
    Builder::EltwiseLayer layerFromNet(net.getLayer(ind));

    ASSERT_EQ(layer.getInputPorts(), layerFromNet.getInputPorts());
    ASSERT_EQ(layer.getOutputPort(), layerFromNet.getOutputPort());
    ASSERT_EQ(layer.getEltwiseType(), layerFromNet.getEltwiseType());
}

TEST_F(EltwiseLayerBuilderTest, checkOnlineEltwiseTypeChanging) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3}), Port({1, 2, 3})});
    layer.setOutputPort(Port({1, 2, 3}));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::MAX);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::MAX);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::DIV);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::DIV);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::MIN);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::MIN);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::MUL);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::MUL);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::SQUARED_DIFF);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::SQUARED_DIFF);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUB);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::SUB);
    ASSERT_NO_THROW(net.addLayer(layer));

    layer.setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM);
    ASSERT_EQ(layer.getEltwiseType(), Builder::EltwiseLayer::EltwiseType::SUM);
    ASSERT_NO_THROW(net.addLayer(layer));
}

TEST_F(EltwiseLayerBuilderTest, cannotCreateLayerWithOneInputPort) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3, 4})});   // here
    layer.setOutputPort(Port({1, 2, 3, 4}));
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(EltwiseLayerBuilderTest, canCreateLayerWithThreeInputPort) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3, 4}), Port({1, 2, 3, 4}), Port({1, 2, 3, 4})});   // here
    layer.setOutputPort(Port({1, 2, 3, 4}));
    ASSERT_NO_THROW(net.addLayer(layer));
}

TEST_F(EltwiseLayerBuilderTest, cannotCreateLayerWithDifferentInputPorts) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3, 4}), Port({1, 2, 3, 1000})});   // here
    layer.setOutputPort(Port({1, 2, 3, 4}));
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(EltwiseLayerBuilderTest, cannotCreateLayerWithDifferentInputAndOutputPorts) {
    Builder::Network net("network");
    Builder::EltwiseLayer layer("Eltwise layer");

    layer.setInputPorts({Port({1, 2, 3, 4}), Port({1, 2, 3, 4})});
    layer.setOutputPort(Port({1, 2, 3, 100}));   // here
    ASSERT_THROW(net.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}
