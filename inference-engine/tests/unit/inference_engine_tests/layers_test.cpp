// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_layers.h>
#include <gtest/gtest.h>
#include <ie_data.h>
#include "ie_precision.hpp"

using namespace std;

class LayersTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

    InferenceEngine::LayerParams getParamsForLayer(std::string name, std::string type,
            InferenceEngine::Precision precision) {
        InferenceEngine::LayerParams params = {};
        params.name = name;
        params.type = type;
        params.precision = precision;
        return params;
    }

    const std::string defaultLayerName = "layer";
    const std::string defaultLayerType = "unknown";
    InferenceEngine::Precision defaultPrecision{InferenceEngine::Precision::FP32};

    InferenceEngine::LayerParams getDefaultParamsForLayer() {
        return getParamsForLayer(defaultLayerName, defaultLayerType, defaultPrecision);
    }

    template <class T>
    bool checkCreateLayer() {
        T layer(getDefaultParamsForLayer());
        return layer.name == defaultLayerName;
    }
public:

};

TEST_F(LayersTests, canCreateLayersWithDefaultParams) {
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::CNNLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ConvolutionLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::DeconvolutionLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::PoolingLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::PowerLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::FullyConnectedLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ConcatLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::SplitLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::NormLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::SoftMaxLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::GRNLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ReLULayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::EltwiseLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::CropLayer>());
    ASSERT_TRUE(checkCreateLayer<InferenceEngine::ScaleShiftLayer>());

}

TEST_F(LayersTests, throwsOnExpiredDataPtr) {
    InferenceEngine::CNNLayer layer(getDefaultParamsForLayer());
    InferenceEngine::DataPtr dataPtr(new InferenceEngine::Data("data", InferenceEngine::Precision::FP32, InferenceEngine::NCHW));
    layer.insData.resize(1);
    layer.insData[0] = dataPtr;
    dataPtr.reset();
    ASSERT_THROW(layer.input(), InferenceEngine::details::InferenceEngineException);
}

