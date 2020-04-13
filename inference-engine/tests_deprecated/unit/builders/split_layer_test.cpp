// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class SplitLayerBuilderTest : public BuilderTestCommon {};

TEST_F(SplitLayerBuilderTest, CreateIdentitySplitLayer) {
    Builder::Network builder("network");
    SizeVector shape = {1, 4, 3, 4};
    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(shape, Precision::FP16)));
    layerId = builder.addLayer({layerId}, Builder::SplitLayer("identity").setOutputPorts({Port()}));
    builder.addLayer({layerId}, Builder::OutputLayer("output"));

    const auto network = builder.build();
    ASSERT_EQ(shape, network->getLayer(layerId)->getOutputPorts()[0].shape());
}

TEST_F(SplitLayerBuilderTest, CreateSplitLayerWithTwoOutputs) {
    Builder::Network builder("network");
    SizeVector shape = {1, 4, 3, 4};
    SizeVector outShape = {1, 2, 3, 4};
    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(shape, Precision::FP16)));
    layerId = builder.addLayer({layerId}, Builder::SplitLayer("split").setOutputPorts({Port(), Port()}));
    builder.addLayer({{layerId}}, Builder::OutputLayer("output1"));
    builder.addLayer({{layerId, 1}}, Builder::OutputLayer("output2"));

    const auto network = builder.build();
    ASSERT_EQ(outShape, network->getLayer(layerId)->getOutputPorts()[0].shape());
    ASSERT_EQ(outShape, network->getLayer(layerId)->getOutputPorts()[1].shape());
}

TEST_F(SplitLayerBuilderTest, CreateSplitLayerWithTwoOutputsAndOneInitialized) {
    Builder::Network builder("network");
    SizeVector shape = {1, 4, 3, 4};
    SizeVector outShape1 = {1, 3, 3, 4};
    SizeVector outShape2 = {1, 1, 3, 4};
    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(shape, Precision::FP16)));
    layerId = builder.addLayer({layerId}, Builder::SplitLayer("split").setOutputPorts({Port(outShape1), Port()}));
    builder.addLayer({{layerId}}, Builder::OutputLayer("output1"));
    builder.addLayer({{layerId, 1}}, Builder::OutputLayer("output2"));

    const auto network = builder.build();
    ASSERT_EQ(outShape1, network->getLayer(layerId)->getOutputPorts()[0].shape());
    ASSERT_EQ(outShape2, network->getLayer(layerId)->getOutputPorts()[1].shape());
}

TEST_F(SplitLayerBuilderTest, CreateSplitLayerWithTwoOutputsAxis3) {
    Builder::Network builder("network");
    SizeVector shape = {1, 4, 3, 4};
    SizeVector outShape = {1, 4, 3, 2};
    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(shape, Precision::FP16)));
    layerId = builder.addLayer({layerId}, Builder::SplitLayer("split").setAxis(3).setOutputPorts({Port(), Port()}));
    builder.addLayer({{layerId}}, Builder::OutputLayer("output1"));
    builder.addLayer({{layerId, 1}}, Builder::OutputLayer("output2"));

    const auto network = builder.build();
    ASSERT_EQ(outShape, network->getLayer(layerId)->getOutputPorts()[0].shape());
    ASSERT_EQ(outShape, network->getLayer(layerId)->getOutputPorts()[1].shape());
}

TEST_F(SplitLayerBuilderTest, CreateSplitLayerWithTwoOutputsAxis3AndOneInitialized) {
    Builder::Network builder("network");
    SizeVector shape = {1, 4, 3, 4};
    SizeVector outShape1 = {1, 4, 3, 1};
    SizeVector outShape2 = {1, 4, 3, 3};
    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(shape, Precision::FP16)));
    layerId = builder.addLayer({layerId}, Builder::SplitLayer("split").setAxis(3).setOutputPorts({Port(outShape1), Port()}));
    builder.addLayer({{layerId}}, Builder::OutputLayer("output1"));
    builder.addLayer({{layerId, 1}}, Builder::OutputLayer("output2"));

    const auto network = builder.build();
    ASSERT_EQ(outShape1, network->getLayer(layerId)->getOutputPorts()[0].shape());
    ASSERT_EQ(outShape2, network->getLayer(layerId)->getOutputPorts()[1].shape());
}