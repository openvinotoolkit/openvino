// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_argmax_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ArgMaxLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ArgMaxLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("network");
    Builder::ArgMaxLayer argMaxLayer("ArgMax layer");
    argMaxLayer.setAxis(1);
    argMaxLayer.setOutMaxVal(0);
    argMaxLayer.setTopK(20);
    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(argMaxLayer));
    Builder::ArgMaxLayer layerFromNetwork(network.getLayer(ind));
    ASSERT_EQ(argMaxLayer.getAxis(), layerFromNetwork.getAxis());
    ASSERT_EQ(argMaxLayer.getOutMaxVal(), layerFromNetwork.getOutMaxVal());
    ASSERT_EQ(argMaxLayer.getTopK(), layerFromNetwork.getTopK());
}

TEST_F(ArgMaxLayerBuilderTest, cannotAddLayerWithWrongAxis) {
    Builder::Network network("network");
    Builder::ArgMaxLayer argMaxLayer("ArgMax layer");
    argMaxLayer.setAxis(500);  // here
    argMaxLayer.setOutMaxVal(0);
    argMaxLayer.setTopK(20);
    ASSERT_THROW(network.addLayer(argMaxLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ArgMaxLayerBuilderTest, cannotAddLayerWithWrongOutMaxVal) {
    Builder::Network network("network");
    Builder::ArgMaxLayer argMaxLayer("ArgMax layer");
    argMaxLayer.setAxis(1);
    argMaxLayer.setOutMaxVal(500);  // here
    argMaxLayer.setTopK(20);
    ASSERT_THROW(network.addLayer(argMaxLayer), InferenceEngine::details::InferenceEngineException);
}