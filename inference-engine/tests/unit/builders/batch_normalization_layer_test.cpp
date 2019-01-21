// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_batch_normalization_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class BatchNormalizationLayerBuilderTest : public BuilderTestCommon {};

TEST_F(BatchNormalizationLayerBuilderTest, cannotCreateBatchNormalizationWithoutWeightOrBiases) {
    ASSERT_THROW(((Builder::Layer)Builder::BatchNormalizationLayer("in1")), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(((Builder::Layer)Builder::BatchNormalizationLayer("in1")
            .setWeights(generateBlob(Precision::FP32, {3}, Layout::C))), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(((Builder::Layer)Builder::BatchNormalizationLayer("in1")
            .setBiases(generateBlob(Precision::FP32, {3}, Layout::C))), InferenceEngine::details::InferenceEngineException);
}

TEST_F(BatchNormalizationLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("Test");
    Builder::BatchNormalizationLayer bnBuilder("bn");
    bnBuilder.setWeights(generateBlob(Precision::FP32, {3}, Layout::C));
    bnBuilder.setBiases(generateBlob(Precision::FP32, {3}, Layout::C));
    size_t bnId = network.addLayer(bnBuilder);
    Builder::BatchNormalizationLayer bnBuilderFromNetwork(network.getLayer(bnId));
    ASSERT_EQ(bnBuilderFromNetwork.getEpsilon(), bnBuilder.getEpsilon());
    bnBuilderFromNetwork.setEpsilon(2);
    ASSERT_NE(bnBuilderFromNetwork.getEpsilon(), bnBuilder.getEpsilon());
    ASSERT_EQ(bnBuilderFromNetwork.getEpsilon(), network.getLayer(bnId).getParameters()["epsilon"].asFloat());
}