// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <transform/transform_network.hpp>
#include <ie_builders.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class TransformNetworkTest: public BuilderTestCommon {};

TEST_F(TransformNetworkTest, AddNewLayer) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    ASSERT_EQ(0, builder.size());
    network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_EQ(1, builder.size());
}

TEST_F(TransformNetworkTest, RemoveLayer) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    ASSERT_EQ(0, builder.size());
    Transform::Layer layer = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_EQ(1, builder.size());

    network.removeLayer(layer);
    ASSERT_EQ(0, builder.size());
}

TEST_F(TransformNetworkTest, GetIncorrectPort) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer layer = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_THROW(layer.getInPort(), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(layer.getOutPort(1), InferenceEngine::details::InferenceEngineException);
}


TEST_F(TransformNetworkTest, GetCorrectPort) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer layer = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_NO_THROW(layer.getOutPort());
    ASSERT_NO_THROW(layer.getOutPort(0));
}

TEST_F(TransformNetworkTest, GetLayerById) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer layer = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_NO_THROW(network.getLayer(layer.getId()));
}

TEST_F(TransformNetworkTest, GetLayerByName) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    ASSERT_NO_THROW(network.getLayer("in1"));
}

TEST_F(TransformNetworkTest, ConnectTwoLayers) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer input = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    Transform::Layer relu = network.addLayer(Builder::ReLULayer("relu1"));
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    network.connect(input, relu);
    ASSERT_EQ(1, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, ConnectTwoPorts) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Port inputPort = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27}))).getOutPort();
    Transform::Port reluPort = network.addLayer(Builder::ReLULayer("relu1")).getInPort();
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    network.connect(inputPort, reluPort);
    ASSERT_EQ(1, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, DisconnectTwoLayers) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer input = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    Transform::Layer relu = network.addLayer(Builder::ReLULayer("relu1"));
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    network.connect(input, relu);
    ASSERT_EQ(1, builder.getConnections().size());
    network.disconnect(input, relu);
    ASSERT_EQ(0, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, DisonnectTwoPorts) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Port inputPort = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27}))).getOutPort();
    Transform::Port reluPort = network.addLayer(Builder::ReLULayer("relu1")).getInPort();
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    network.connect(inputPort, reluPort);
    ASSERT_EQ(1, builder.getConnections().size());
    network.disconnect(inputPort, reluPort);
    ASSERT_EQ(0, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, RemoveLayerAndConnection) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer input = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    Transform::Layer relu = network.addLayer(Builder::ReLULayer("relu1"));
    network.connect(input, relu);
    ASSERT_EQ(1, builder.getConnections().size());
    ASSERT_EQ(2, builder.size());
    network.removeLayer(relu);
    ASSERT_EQ(0, builder.getConnections().size());
    ASSERT_EQ(1, builder.size());
}

TEST_F(TransformNetworkTest, GetInitializedConnection) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer input = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    Transform::Layer relu = network.addLayer(Builder::ReLULayer("relu1"));
    network.connect(input, relu);
    ASSERT_EQ(input.getOutPort(), relu.getInPort().getConnection().getSource());
}

TEST_F(TransformNetworkTest, GetIncorrectConnections) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Layer input = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27})));
    Transform::Layer relu = network.addLayer(Builder::ReLULayer("relu1"));
    ASSERT_THROW(relu.getInPort().getConnection().getSource(), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(input.getOutPort().getConnection().getDestination(), InferenceEngine::details::InferenceEngineException);
    ASSERT_NO_THROW(input.getOutPort().getConnection().getSource());
    ASSERT_NO_THROW(relu.getInPort().getConnection().getDestination());
}

TEST_F(TransformNetworkTest, ConnectToSourcePortsFromConnection) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Port inputPort = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27}))).getOutPort();
    Transform::Port reluPort = network.addLayer(Builder::ReLULayer("relu1")).getInPort();
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    ASSERT_NO_THROW(inputPort.getConnection().setDestination(reluPort));
    ASSERT_EQ(1, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, ConnectWithTwoDestinations) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Port inputPort = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27}))).getOutPort();
    Transform::Port reluPort1 = network.addLayer(Builder::ReLULayer("relu1")).getInPort();
    Transform::Port reluPort2 = network.addLayer(Builder::ReLULayer("relu2")).getInPort();
    ASSERT_EQ(3, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    ASSERT_NO_THROW(inputPort.getConnection().setDestination(reluPort1));
    ASSERT_NO_THROW(inputPort.getConnection().addDestination(reluPort2));
    ASSERT_THROW(inputPort.getConnection().addDestination(reluPort2), InferenceEngine::details::InferenceEngineException);
    ASSERT_EQ(2, builder.getConnections().size());
    ASSERT_THROW(inputPort.getConnection().setDestination(reluPort2), InferenceEngine::details::InferenceEngineException);
    ASSERT_NO_THROW(inputPort.getConnection().setDestinations({reluPort2, reluPort1}));
    ASSERT_EQ(2, builder.getConnections().size());
}

TEST_F(TransformNetworkTest, ConnectToDestinationPortsFromConnection) {
    Builder::Network builder("test");
    Transform::Network network(builder);
    Transform::Port inputPort = network.addLayer(Builder::InputLayer("in1").setPort(Port({1, 3, 27, 27}))).getOutPort();
    Transform::Port reluPort = network.addLayer(Builder::ReLULayer("relu1")).getInPort();
    ASSERT_EQ(2, builder.size());
    ASSERT_EQ(0, builder.getConnections().size());
    reluPort.getConnection().setSource(inputPort);
    ASSERT_EQ(1, builder.getConnections().size());
}