// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_deconvolution_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class DeconvolutionLayerBuilderTest : public BuilderTestCommon {};

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithoutWeight) {
    Builder::Network network("Test");

    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");
    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));
    deconvBuilder.setDilation({1, 1});
    size_t ind = network.addLayer(deconvBuilder);
    ASSERT_THROW(network.getLayer(ind)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, getExistsLayerFromNetworkBuilderWithInputPort) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));
    deconvBuilder.setDilation({1, 1});

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    Builder::DeconvolutionLayer deconvBuilderFromNetwork(network.getLayer(convId));

    ASSERT_EQ(deconvBuilderFromNetwork.getStrides(), deconvBuilder.getStrides());
    ASSERT_EQ(deconvBuilderFromNetwork.getKernel(), deconvBuilder.getKernel());
    ASSERT_EQ(deconvBuilderFromNetwork.getPaddingsEnd(), deconvBuilder.getPaddingsEnd());
    ASSERT_EQ(deconvBuilderFromNetwork.getPaddingsBegin(), deconvBuilder.getPaddingsBegin());
    ASSERT_EQ(deconvBuilderFromNetwork.getOutDepth(), deconvBuilder.getOutDepth());
    ASSERT_EQ(deconvBuilderFromNetwork.getDilation(), deconvBuilder.getDilation());
}

TEST_F(DeconvolutionLayerBuilderTest, getExistsLayerFromNetworkBuilderWithoutInputPort) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setDilation({1, 1});

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    Builder::DeconvolutionLayer deconvBuilderFromNetwork(network.getLayer(convId));

    ASSERT_EQ(deconvBuilderFromNetwork.getStrides(), deconvBuilder.getStrides());
    ASSERT_EQ(deconvBuilderFromNetwork.getKernel(), deconvBuilder.getKernel());
    ASSERT_EQ(deconvBuilderFromNetwork.getPaddingsEnd(), deconvBuilder.getPaddingsEnd());
    ASSERT_EQ(deconvBuilderFromNetwork.getPaddingsBegin(), deconvBuilder.getPaddingsBegin());
    ASSERT_EQ(deconvBuilderFromNetwork.getOutDepth(), deconvBuilder.getOutDepth());
    ASSERT_EQ(deconvBuilderFromNetwork.getDilation(), deconvBuilder.getDilation());
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 64, 225, 225}));  // here

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, canCreateCorrcetConvolution) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));  // here

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithGroup) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setGroup(2);
    deconvBuilder.setInputPort(Port({1, 6, 225, 225}));

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 6, 11, 11}, Layout::OIHW)));
    // should be {96, 6 / 2, 11, 11}
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, canCreateConvolution) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setGroup(2);
    deconvBuilder.setInputPort(Port({1, 6, 225, 225}));  // here

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongOutDepth) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(4);  // here
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));

    idx_t convId = network.addLayer(deconvBuilder);

    idx_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 3, 11, 11}, Layout::OIHW)));
    network.connect({weightsId}, {convId, 1});

    idx_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));
    network.connect({biasesId}, {convId, 2});

    ASSERT_THROW(network.getLayer(convId)->validate(false), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongStrides) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 0});  // here
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));
    deconvBuilder.setPaddingsEnd({0, 0});
    deconvBuilder.setPaddingsBegin({0, 0});
    deconvBuilder.setDilation({0, 0});
    ASSERT_THROW(network.addLayer(deconvBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongKernel1) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 0});  // here
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));

    ASSERT_THROW(network.addLayer(deconvBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongKernel2) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer convBuilder("Deconvolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11, 11});  // here
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongDilation1) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 3, 225, 225}));
    deconvBuilder.setDilation({1, 0});  // here

    ASSERT_THROW(network.addLayer(deconvBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateConvolutionWithWrongDilation2) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer convBuilder("Deconvolution");

    convBuilder.setStrides({4, 4});
    convBuilder.setKernel({11, 11});
    convBuilder.setOutDepth(96);
    convBuilder.setInputPort(Port({1, 3, 225, 225}));
    convBuilder.setDilation({1, 1, 1});  // here

    ASSERT_THROW(network.addLayer(convBuilder), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeconvolutionLayerBuilderTest, canCreateLayerWithNumberOfGroupDividingNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 2, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 6, 225, 225}));
    deconvBuilder.setDilation({1, 1});

    deconvBuilder.setGroup(3);
    size_t convId = network.addLayer(deconvBuilder);
    network.connect({weightsId}, {convId, 1});
    network.connect({biasesId}, {convId, 2});
    ASSERT_NO_THROW(network.getLayer(convId)->validate(false));
}

TEST_F(DeconvolutionLayerBuilderTest, canCreateLayerWithWeightsNotAvailableForGroup) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 5, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 6, 225, 225}));
    deconvBuilder.setDilation({1, 1});

    deconvBuilder.setGroup(3);
    ASSERT_THROW(network.addLayer({{weightsId}, {biasesId}}, deconvBuilder),
                 InferenceEngine::details::InferenceEngineException);  // 6 / 3 != 5
}

TEST_F(DeconvolutionLayerBuilderTest, cannotCreateLayerWithNumberOfGroupNotDividingNumberOfInputChannels) {
    Builder::Network network("Test");
    Builder::DeconvolutionLayer deconvBuilder("Deconvolution");

    size_t weightsId = network.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP32, {96, 2, 11, 11}, Layout::OIHW)));
    size_t biasesId = network.addLayer(Builder::ConstLayer("biases").setData(generateBlob(Precision::FP32, {96}, Layout::C)));

    deconvBuilder.setStrides({4, 4});
    deconvBuilder.setKernel({11, 11});
    deconvBuilder.setOutDepth(96);
    deconvBuilder.setInputPort(Port({1, 6, 225, 225}));
    deconvBuilder.setDilation({1, 1});

    deconvBuilder.setGroup(4);
    ASSERT_THROW(network.addLayer({{weightsId}, {biasesId}}, deconvBuilder),
                 InferenceEngine::details::InferenceEngineException);  // 6 % 4 == 2
}