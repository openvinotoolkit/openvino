// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <string.h>
#include <ie_builders.hpp>
#include <builders/ie_detection_output_layer.hpp>

#include "builder_test.hpp"

using namespace testing;
using namespace InferenceEngine;

class DetectionOutputLayerBuilderTest : public BuilderTestCommon {};

TEST_F(DetectionOutputLayerBuilderTest, getExistsLayerFromNetworkBuilder) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(2);
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-1);
    layer.setNMSThreshold(0.45);
    layer.setTopK(400);
    layer.setCodeType("caffe.PriorBoxParameter.CENTER_SIZE");
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(0.01);
    size_t ind = 0;
    ASSERT_NO_THROW(ind = network.addLayer(layer));
    Builder::DetectionOutputLayer layerFromNet(network.getLayer(ind));
    ASSERT_EQ(layerFromNet.getName(), layer.getName());
    ASSERT_EQ(layerFromNet.getNumClasses(), layer.getNumClasses());
    ASSERT_EQ(layerFromNet.getShareLocation(), layer.getShareLocation());
    ASSERT_EQ(layerFromNet.getBackgroudLabelId(), layer.getBackgroudLabelId());
    ASSERT_EQ(layerFromNet.getNMSThreshold(), layer.getNMSThreshold());
    ASSERT_EQ(layerFromNet.getTopK(), layer.getTopK());
    ASSERT_EQ(layerFromNet.getCodeType(), layer.getCodeType());
    ASSERT_EQ(layerFromNet.getVariantEncodedInTarget(), layer.getVariantEncodedInTarget());
    ASSERT_EQ(layerFromNet.getKeepTopK(), layer.getKeepTopK());
    ASSERT_EQ(layerFromNet.getConfidenceThreshold(), layer.getConfidenceThreshold());
}

TEST_F(DetectionOutputLayerBuilderTest, cannotCreateLayerWithWrongNumClasses) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(0);  // here
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-1);
    layer.setNMSThreshold(0.45);
    layer.setTopK(400);
    layer.setCodeType("caffe.PriorBoxParameter.CENTER_SIZE");
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(0.01);
    ASSERT_THROW(network.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DetectionOutputLayerBuilderTest, cannotCreateLayerWithWrongCodeType) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(2);
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-1);
    layer.setNMSThreshold(0.45);
    layer.setTopK(400);
    layer.setCodeType("trololo");  // here
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(0.01);
    ASSERT_THROW(network.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DetectionOutputLayerBuilderTest, cannotCreateLayerWithWrongBackLabelId) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(2);
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-100);  // here
    layer.setNMSThreshold(0.45);
    layer.setTopK(400);
    layer.setCodeType("caffe.PriorBoxParameter.CENTER_SIZE");
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(0.01);
    ASSERT_THROW(network.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DetectionOutputLayerBuilderTest, cannotCreateLayerWithWrongNMSThreshold) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(2);
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-1);
    layer.setNMSThreshold(-0.02);  // here
    layer.setTopK(400);
    layer.setCodeType("caffe.PriorBoxParameter.CENTER_SIZE");
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(0.01);
    ASSERT_THROW(network.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DetectionOutputLayerBuilderTest, cannotCreateLayerWithWrongConfidenceThreshold) {
    Builder::Network network("network");
    Builder::DetectionOutputLayer layer("detection output layer");
    layer.setNumClasses(2);
    layer.setShareLocation(true);
    layer.setBackgroudLabelId(-1);
    layer.setNMSThreshold(0.45);
    layer.setTopK(400);
    layer.setCodeType("caffe.PriorBoxParameter.CENTER_SIZE");
    layer.setVariantEncodedInTarget(false);
    layer.setKeepTopK(200);
    layer.setConfidenceThreshold(-0.1);  // here
    ASSERT_THROW(network.addLayer(layer), InferenceEngine::details::InferenceEngineException);
}
