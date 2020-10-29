// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_convolution_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class ConvolutionLayerBuilderTest : public BuilderTestCommon {};

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithoutWeight) {
    Builder::Network network("Test");

    Builder::ConvolutionLayer convBuilder("Convolution");
    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setDilation({1, 1});
    size_t ind = network.addLayer(convBuilder);
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, getExistsLayerFromNetworkBuilderWithInputPort) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setDilation({1, 1});

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    Builder::ConvolutionLayer convBuilderFromNetwork(network.getLayer(convId));

    ASSERT_EQ(convBuilderFromNetwork.getStrides(), convBuilder.getStrides());
    ASSERT_EQ(convBuilderFromNetwork.getKernel(), convBuilder.getKernel());
    ASSERT_EQ(convBuilderFromNetwork.getPaddingsEnd(), convBuilder.getPaddingsEnd());
    ASSERT_EQ(convBuilderFromNetwork.getPaddingsBegin(), convBuilder.getPaddingsBegin());
    ASSERT_EQ(convBuilderFromNetwork.getOutDepth(), convBuilder.getOutDepth());
    ASSERT_EQ(convBuilderFromNetwork.getDilation(), convBuilder.getDilation());
}

TEST_F(ConvolutionLayerBuilderTest, getExistsLayerFromNetworkBuilderWithoutInputPort) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setDilation({1, 1});

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    Builder::ConvolutionLayer convBuilderFromNetwork(network.getLayer(convId));

    ASSERT_EQ(convBuilderFromNetwork.getStrides(), convBuilder.getStrides());
    ASSERT_EQ(convBuilderFromNetwork.getKernel(), convBuilder.getKernel());
    ASSERT_EQ(convBuilderFromNetwork.getPaddingsEnd(), convBuilder.getPaddingsEnd());
    ASSERT_EQ(convBuilderFromNetwork.getPaddingsBegin(), convBuilder.getPaddingsBegin());
    ASSERT_EQ(convBuilderFromNetwork.getOutDepth(), convBuilder.getOutDepth());
    ASSERT_EQ(convBuilderFromNetwork.getDilation(), convBuilder.getDilation());
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 64, 225, 225}));  // here

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, canCreateCorrcetConvolution) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));  // here

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithGroup) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setGroup(2);
    convBuilder.setInputPort(Port({1, 6, 225, 225}));

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 6, 11, 11}, Layout::OIHW)));
    // should be {96, 6 / 2, 11, 11}
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, canCreateConvolution) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setGroup(2);
    convBuilder.setInputPort(Port({1, 6, 225, 225}));  // here

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongOutDepth) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(4);  // here
    convBuilder.setInputPort(Port({1, 3, 225, 225}));

    idx_t convId = network.addLayer(convBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongStrides) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 0});  // here
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setPaddingsEnd({0, 0});
    convBuilder.setPaddingsBegin({0, 0});
    convBuilder.setDilation({0, 0});
    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongKernel1) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 0});  // here
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongKernel2) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11, 11});  // here
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongDilation1) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setDilation({1, 0});  // here

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongDilation2) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convBuilder("Convolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setDilation({1, 1, 1});  // here

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(ConvolutionLayerBuilderTest, canCreateLayerWithNumberOfGroupDividingNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convLayer("Convolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 2, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    convLayer.setStrides({4, 4});
    convLayer.setKernel({11, 11});
    convLayer.setOutDepth(96);
    convLayer.setInputPort(Port({1, 6, 225, 225}));
    convLayer.setDilation({1, 1});

    convLayer.setGroup(3);
    size_t convId = network.addLayer(convLayer);
    network.connect({weightsId}, {convId, 1});
    network.connect({biasesId}, {convId, 2});
    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(ConvolutionLayerBuilderTest, canCreateLayerWithWeightsNotAvailableForGroup) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convLayer("Convolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 5, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    convLayer.setStrides({4, 4});
    convLayer.setKernel({11, 11});
    convLayer.setOutDepth(96);
    convLayer.setInputPort(Port({1, 6, 225, 225}));
    convLayer.setDilation({1, 1});

    convLayer.setGroup(3);
    ASSERT_THROW(network.addLayer({{weightsId}, {biasesId}}, convLayer),
                 InferenceEngine::details::InferenceEngineException);  // 6 / 3 != 5
}

TEST_F(ConvolutionLayerBuilderTest, cannotCreateLayerWithNumberOfGroupNotDividingNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::ConvolutionLayer convLayer("Convolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 2, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    convLayer.setStrides({4, 4});
    convLayer.setKernel({11, 11});
    convLayer.setOutDepth(96);
    convLayer.setInputPort(Port({1, 6, 225, 225}));
    convLayer.setDilation({1, 1});

    convLayer.setGroup(4);
    ASSERT_THROW(network.addLayer({{weightsId}, {biasesId}}, convLayer),
                 InferenceEngine::details::InferenceEngineException);  // 6 % 4 == 2
}

