// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class MemoryLayerBuilderTest : public BuilderTestCommon {};


TEST_F(MemoryLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("Test");
    Builder::MemoryLayer memoryInBuilder("MemoryIn1"), memoryOutBuilder("MemoryOut1");
    Builder::ConcatLayer concat("concat");
    Builder::InputLayer input("inLayer");
    Builder::FullyConnectedLayer fc("fc0");

    memoryInBuilder.setOutputPort(Port({1, 30}));
    memoryOutBuilder.setInputPort(Port({1, 30}));

    input.setPort(Port({1, 30}));
    concat.setInputPorts({Port({1,30}), Port({1, 30})});
    concat.setOutputPort(Port({1, 60}));
    fc.setInputPort(Port({1, 60}));
    fc.setOutputPort(Port({1, 30}));

    size_t inId  = network.addLayer(memoryInBuilder);
    size_t outId  = network.addLayer(memoryOutBuilder);
    size_t inId2  = network.addLayer(concat);
    size_t inId3  = network.addLayer(input);
    size_t inIdfc = network.addLayer(fc);

    network.connect({inId3}, {inId2, 0});
    network.connect({inId}, {inId2, 1});
    network.connect({inId2}, {inIdfc});
    network.connect({inIdfc}, {outId});


    ASSERT_EQ(memoryInBuilder.getOutputPort().shape(), Port({1, 30}).shape());
    auto cnn_network = Builder::convertToICNNNetwork(network.build());

    CNNLayerPtr layer;
    cnn_network->getLayerByName("concat", layer, nullptr);
    ASSERT_EQ(layer->outData.size(), 1);
}