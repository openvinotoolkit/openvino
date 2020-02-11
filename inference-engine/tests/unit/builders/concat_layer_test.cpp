// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_concat_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ConcatLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ConcatLayerBuilderTest, getExistsLayerFromNetworkBuilderAxis) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({1 + 3, 2, 55, 55}));

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    network.getLayer(ind)->validate(false);
    ASSERT_NO_THROW(network.getLayer(ind)->validate(false));
    Builder::ConcatLayer layerFromNet(network.getLayer(ind));

    ASSERT_EQ(layer.getAxis(), layerFromNet.getAxis());
    ASSERT_EQ(layer.getInputPorts(), layerFromNet.getInputPorts());
    ASSERT_EQ(layer.getOutputPort(), layerFromNet.getOutputPort());
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithNoInputPorts) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(1);
    layer.setOutputPort(Port({1, 2 + 4, 55, 55}));
    // here should be layer.setInputPort(...)

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithOneInputPort) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(1);
    layer.setInputPorts({Port({1, 2, 55, 55})});  // here
    layer.setOutputPort(Port({1, 2 + 4, 55, 55}));

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithWrongAxis) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(50);  // here
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({1 + 3, 2, 55, 55}));

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithUnalignedPorts1) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({1 + 3, 2, 55, 155}));  // should be {1 + 3, 2, 55, 55}

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithUnalignedPorts2) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({1 + 3, 2, 155, 55}));  // should be {1 + 3, 2, 55, 55}

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithUnalignedPorts3) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({100, 2, 55, 55}));  // should be {1 + 3, 2, 55, 55}

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithUnalignedPorts4) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(1);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 55})});
    layer.setOutputPort(Port({1, 100, 55, 55}));  // should be {1, 2 + 4, 55, 55}

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithDifferentInputPorts1) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 55, 155})});  // here
    layer.setOutputPort(Port({1 + 3, 4, 55, 55}));

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConcatLayerBuilderTest, cannotCreateLayerWithDifferentInputPorts2) {
    Builder::Network network("network");
    Builder::ConcatLayer layer("concat layer");

    layer.setAxis(0);
    layer.setInputPorts({Port({1, 2, 55, 55}), Port({3, 2, 155, 55})});  // here
    layer.setOutputPort(Port({1 + 3, 4, 55, 55}));

    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}