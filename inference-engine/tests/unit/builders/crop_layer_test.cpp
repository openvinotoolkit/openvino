// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_crop_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class CropLayerBuilderTest : public BuilderTestCommon {};

TEST_F(CropLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("network");
    Builder::CropLayer cropLayer("Crop layer");
    std::vector<Port> input_ports;
    input_ports.push_back(Port({1, 21, 44, 44}));
    input_ports.push_back(Port({1, 21, 44, 44}));
    cropLayer.setInputPorts(input_ports);
    cropLayer.setOutputPort(Port({1, 21, 44, 44}));
    cropLayer.setAxis({2, 3});
    cropLayer.setOffset({0, 0});
    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(cropLayer));
    Builder::CropLayer layerFromNet(network.getLayer(ind));
    ASSERT_EQ(layerFromNet.getAxis(), cropLayer.getAxis());
    ASSERT_EQ(layerFromNet.getOffset(), cropLayer.getOffset());
}

TEST_F(CropLayerBuilderTest, cannotCreateLayerWithOneInputShape) {
    Builder::Network network("network");
    Builder::CropLayer cropLayer("Crop layer");
    std::vector<Port> input_ports;
    input_ports.push_back(Port({1, 21, 44, 44}));  // here
    cropLayer.setInputPorts(input_ports);
    cropLayer.setOutputPort(Port({1, 21, 44, 44}));
    cropLayer.setAxis({2, 3});
    cropLayer.setOffset({0, 0});
    ASSERT_THROW(network.addLayer(cropLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CropLayerBuilderTest, cannotCreateLayerWithThreeInputShapes) {
    Builder::Network network("network");
    Builder::CropLayer cropLayer("Crop layer");
    std::vector<Port> input_ports;
    input_ports.push_back(Port({1, 21, 44, 44}));
    input_ports.push_back(Port({1, 21, 44, 44}));
    input_ports.push_back(Port({1, 21, 44, 44}));  // here
    cropLayer.setInputPorts(input_ports);
    cropLayer.setOutputPort(Port({1, 21, 44, 44}));
    cropLayer.setAxis({2, 3});
    cropLayer.setOffset({0, 0});
    ASSERT_THROW(network.addLayer(cropLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CropLayerBuilderTest, cannotCreateLayerWithDifferentSizeOfAxisAndOffset) {
    Builder::Network network("network");
    Builder::CropLayer cropLayer("Crop layer");
    std::vector<Port> input_ports;
    input_ports.push_back(Port({1, 21, 44, 44}));
    input_ports.push_back(Port({1, 21, 44, 44}));
    cropLayer.setInputPorts(input_ports);
    cropLayer.setOutputPort(Port({1, 21, 44, 44}));
    cropLayer.setAxis({2, 3});
    cropLayer.setOffset({0, 0, 0});  // here
    ASSERT_THROW(network.addLayer(cropLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CropLayerBuilderTest, cannotCreateLayerWithSoBigOffset) {
    Builder::Network network("network");
    Builder::CropLayer cropLayer("Crop layer");
    std::vector<Port> input_ports;
    input_ports.push_back(Port({1, 21, 44, 44}));
    input_ports.push_back(Port({1, 21, 34, 34}));
    cropLayer.setInputPorts(input_ports);
    cropLayer.setOutputPort(Port({1, 21, 34, 34}));
    cropLayer.setAxis({2, 3});
    cropLayer.setOffset({0, 50});  // here
    ASSERT_THROW(network.addLayer(cropLayer), InferenceEngine::details::InferenceEngineException);
}