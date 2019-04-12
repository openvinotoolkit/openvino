// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_output_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class OutputLayerBuilderTest : public BuilderTestCommon {};

TEST_F(OutputLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("network");
    Builder::OutputLayer layer("output layer");
    layer.setPort(Port({1, 1, 1, 1}));
    size_t ind = network.addLayer(layer);
    Builder::OutputLayer layerFromNet(network.getLayer(ind));
    ASSERT_EQ(layer.getPort().shape(), layerFromNet.getPort().shape());
    ASSERT_EQ(layer.getPort().shape(), Port({1, 1, 1, 1}).shape());
}