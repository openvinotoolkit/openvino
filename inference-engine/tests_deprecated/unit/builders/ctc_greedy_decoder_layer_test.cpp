// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_ctc_greedy_decoder_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class CTCGreedyDecoderLayerBuilderTest : public BuilderTestCommon {};

TEST_F(CTCGreedyDecoderLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("network");
    Builder::CTCGreedyDecoderLayer ctcGreedyDecoderLayer("CTCGreedyDecoder");
    ctcGreedyDecoderLayer.setInputPorts({Port({88, 1, 71}), Port({88, 1})});
    ctcGreedyDecoderLayer.setOutputPort(Port({1, 88, 1, 1}));
    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(ctcGreedyDecoderLayer));
    Builder::CTCGreedyDecoderLayer layerFromNet(network.getLayer(ind));
    ASSERT_EQ(ctcGreedyDecoderLayer.getInputPorts(), layerFromNet.getInputPorts());
    ASSERT_EQ(ctcGreedyDecoderLayer.getOutputPort(), layerFromNet.getOutputPort());
}

TEST_F(CTCGreedyDecoderLayerBuilderTest, cannotCreateLayerWithoutInputPorts) {
    Builder::Network network("network");
    Builder::CTCGreedyDecoderLayer ctcGreedyDecoderLayer("CTCGreedyDecoder");
    ctcGreedyDecoderLayer.setOutputPort(Port({1, 88, 1, 1}));
    ASSERT_THROW(network.addLayer(ctcGreedyDecoderLayer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CTCGreedyDecoderLayerBuilderTest, cannotCreateLayerWithThreeInputPorts) {
    Builder::Network network("network");
    Builder::CTCGreedyDecoderLayer ctcGreedyDecoderLayer("CTCGreedyDecoder");
    ctcGreedyDecoderLayer.setInputPorts({Port({88, 1, 71}), Port({88, 1}), Port({88, 1})});
    ctcGreedyDecoderLayer.setOutputPort(Port({1, 88, 1, 1}));
    ASSERT_THROW(network.addLayer(ctcGreedyDecoderLayer), InferenceEngine::details::InferenceEngineException);
}