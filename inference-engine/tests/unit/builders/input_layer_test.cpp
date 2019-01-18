// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class InputLayerBuilderTest : public BuilderTestCommon {};

TEST_F(InputLayerBuilderTest, cannotCreateInputWithoutPort) {
    ASSERT_THROW(((Builder::Layer)Builder::InputLayer("in1")).build(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(InputLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("Test");
    Builder::InputLayer inBuilder("in1");
    inBuilder.setPort(Port({1, 3, 3, 3}));
    size_t inId = network.addLayer(inBuilder);
    ASSERT_EQ(inBuilder.getPort().shape(), Port({1, 3, 3, 3}).shape());
    Builder::InputLayer inBuilderFromNetwork(network.getLayer(inId));
    ASSERT_EQ(inBuilderFromNetwork.getPort().shape(), Port({1, 3, 3, 3}).shape());
    inBuilderFromNetwork.setPort(Port({1, 3, 4, 4}));
    ASSERT_EQ(inBuilderFromNetwork.getPort().shape(), Port({1, 3, 4, 4}).shape());
    ASSERT_EQ(network.getLayer(inId).getOutputPorts()[0].shape(), Port({1, 3, 4, 4}).shape());
    ASSERT_EQ(inBuilder.getPort().shape(), Port({1, 3, 3, 3}).shape());
}