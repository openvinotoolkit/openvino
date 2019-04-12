// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_const_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ConstLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ConstLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network net("network");
    Builder::ConstLayer layer("const layer");
    layer.setData(generateBlob(Precision::FP32, {3}, Layout::C));
    const size_t ind = net.addLayer(layer);
    ASSERT_NO_THROW(net.getLayer(ind)->validate(false));
}

TEST_F(ConstLayerBuilderTest, cannotCreateLayerWithoutData) {
    Builder::Network net("network");
    Builder::ConstLayer layer("const layer");
    ASSERT_THROW(net.addLayer(layer),
            InferenceEngine::details::InferenceEngineException);
}