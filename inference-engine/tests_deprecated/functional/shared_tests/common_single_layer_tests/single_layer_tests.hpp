// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <tests_common.hpp>
#include <legacy/ie_layers_internal.hpp>
#include <legacy/details/ie_cnn_network_iterator.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include "single_layer_common.hpp"

#include "conv_ref.hpp"
#include "deconv_ref.hpp"
#include "def_conv_ref.hpp"
#include "pool_ref.hpp"
#include "single_layer_common.hpp"
#include "common_layers_params.hpp"
#include <xml_net_builder.hpp>

using namespace InferenceEngine;

struct PluginDependentParam {
    std::string deviceName;
    InferenceEngine::Layout layout;
    InferenceEngine::Precision precision;
    float tolerance;
};

class LayerTestHelper {
protected:
    std::string type;
public:
    using Ptr = std::shared_ptr<LayerTestHelper>;
    explicit LayerTestHelper(const std::string &_type) : type(_type) {}

    virtual ~LayerTestHelper() = default; 
    LayerTestHelper() = default;

    virtual void updatePaddingValues(const InferenceEngine::CNNNetwork &network) = 0;

    virtual std::map<std::string, std::string> getMapParams() const = 0;

    virtual size_t getWeightByteSize(size_t elementSize, size_t numChannels) const = 0;

    virtual size_t getBiasByteSize(size_t elementSize) const = 0;

    std::string getType() const { return type; }

    virtual void ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                          InferenceEngine::Blob &dst,
                          const float *weights_data,
                          size_t weights_size,
                          const float *bias_data,
                          size_t bias_size) const = 0;

    virtual void ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs,
                          InferenceEngine::Blob &dst,
                          const InferenceEngine::ie_fp16 *weights_data,
                          size_t weights_size,
                          const InferenceEngine::ie_fp16 *bias_data,
                          size_t bias_size) const = 0;

    InferenceEngine::Blob::Ptr getRefBlob(size_t weightSize, size_t biasSize,
                                          const InferenceEngine::TBlob<uint8_t>::Ptr &weights,
                                          const std::vector<InferenceEngine::Blob::Ptr> srcs,
                                          const InferenceEngine::TensorDesc &dstTensorDesc,
                                          const InferenceEngine::Precision &precision) const;

    static std::string propertyToString(const InferenceEngine::PropertyVector<unsigned int> &propertyVector);
};

class ConvolutionTestHelper : public LayerTestHelper {
protected:
    CommonTestUtils::conv_common_params convParams;
public:
    explicit ConvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams);

    void updatePaddingValues(const InferenceEngine::CNNNetwork &network) override;

    std::map<std::string, std::string> getMapParams() const override;

    size_t getWeightByteSize(size_t elementSize, size_t numChannels) const override;

    size_t getBiasByteSize(size_t elementSize) const override;

    void ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const float *weights_data,
                  size_t weights_size, const float *bias_data, size_t bias_size) const override;

    void
    ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const InferenceEngine::ie_fp16 *weights_data,
             size_t weights_size, const InferenceEngine::ie_fp16 *bias_data, size_t bias_size) const override;
};

class DeconvolutionTestHelper : public ConvolutionTestHelper {
public:
    explicit DeconvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams);

    void ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const float *weights_data,
                  size_t weights_size, const float *bias_data, size_t bias_size) const override;

    void
    ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const InferenceEngine::ie_fp16 *weights_data,
             size_t weights_size, const InferenceEngine::ie_fp16 *bias_data, size_t bias_size) const override;
};

class DeformableConvolutionTestHelper : public ConvolutionTestHelper {
protected:
    CommonTestUtils::def_conv_common_params defConvParams;
public:
    explicit DeformableConvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams, const int deformable_group);

    void updatePaddingValues(const InferenceEngine::CNNNetwork &network) override;

    std::map<std::string, std::string> getMapParams() const override;

    void ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const float *weights_data,
                  size_t weights_size, const float *bias_data, size_t bias_size) const override;

    void
    ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const InferenceEngine::ie_fp16 *weights_data,
             size_t weights_size, const InferenceEngine::ie_fp16 *bias_data, size_t bias_size) const override;
};

class PoolingTestHelper : public LayerTestHelper {
protected:
    CommonTestUtils::pool_common_params poolParams;
public:
    explicit PoolingTestHelper(const CommonTestUtils::pool_common_params &_poolParams);

    void ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const float *weights_data,
                  size_t weights_size, const float *bias_data, size_t bias_size) const override;

    void
    ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, InferenceEngine::Blob &dst, const InferenceEngine::ie_fp16 *weights_data,
             size_t weights_size, const InferenceEngine::ie_fp16 *bias_data, size_t bias_size) const override;

    std::map<std::string, std::string> getMapParams() const override;

    void updatePaddingValues(const InferenceEngine::CNNNetwork &network) override;

    size_t getWeightByteSize(size_t elementSize, size_t numChannels) const override;

    size_t getBiasByteSize(size_t elementSize) const override;
};

PRETTY_PARAM(InitialShapes, CommonTestUtils::InOutShapes)

PRETTY_PARAM(NewShapes, CommonTestUtils::InOutShapes)

PRETTY_PARAM(ConvParams, CommonTestUtils::conv_common_params)

PRETTY_PARAM(PluginParams, PluginDependentParam)

PRETTY_PARAM(Helper, LayerTestHelper::Ptr)

Blob::Ptr LayerTestHelper::getRefBlob(size_t weightSize, size_t biasSize,
                                      const TBlob<uint8_t>::Ptr &weights,
                                      const std::vector<InferenceEngine::Blob::Ptr> srcs,
                                      const TensorDesc &dstTensorDesc,
                                      const Precision &precision) const {
    Blob::Ptr dst_ref;
    if (precision == Precision::FP32) {
        dst_ref = make_shared_blob<float>(dstTensorDesc);
        dst_ref->allocate();
        const auto *weights_data = weights->buffer().as<const float *>();
        ref_fp32(srcs, *dst_ref.get(), weights_data, weightSize, weights_data + weightSize, biasSize);
    } else {
        dst_ref = make_shared_blob<ie_fp16>(dstTensorDesc);
        dst_ref->allocate();
        const auto *weights_data = weights->buffer().as<const ie_fp16 *>();
        ref_fp16(srcs, *dst_ref.get(), weights_data, weightSize, weights_data + weightSize, biasSize);
    }
    return dst_ref;
}

std::string LayerTestHelper::propertyToString(const PropertyVector<unsigned int> &propertyVector) {
    if (!propertyVector.size()) return "";
    std::string result = std::to_string(propertyVector[0]);
    for (int i = 1; i < propertyVector.size(); i++) {
        result += "," + std::to_string(propertyVector[i]);
    }
    return result;
}

ConvolutionTestHelper::ConvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams) : LayerTestHelper("Convolution"), convParams(_convParams) {}

void ConvolutionTestHelper::updatePaddingValues(const CNNNetwork &network) {
    details::CNNNetworkIterator i(network), end;
    auto found = std::find_if(i, end, [this](const CNNLayer::Ptr &layer) {
        return layer->type == type;
    });
    ASSERT_NE(found, end);

    auto castedLayer = std::dynamic_pointer_cast<ConvolutionLayer>(*found);
    auto allPad = getPaddings(*castedLayer.get());
    convParams.pads_end = allPad.end;
    convParams.pads_begin = allPad.begin;
}

std::map<std::string, std::string> ConvolutionTestHelper::getMapParams() const {
    std::map<std::string, std::string> params;
    if (!convParams.auto_pad.empty()) {
        params["auto_pad"] = convParams.auto_pad;
    }
    params["group"] = std::to_string(convParams.group);
    params["output"] = std::to_string(convParams.out_c);

    auto propertyToString = [](const PropertyVector<unsigned int> &propertyVector) -> std::string {
        if (!propertyVector.size()) return "";
        std::string result = std::to_string(propertyVector[0]);
        for (int i = 1; i < propertyVector.size(); i++) {
            result += "," + std::to_string(propertyVector[i]);
        }
        return result;
    };
    params["kernel"] = propertyToString(convParams.kernel);
    params["strides"] = propertyToString(convParams.stride);
    params["pads_begin"] = propertyToString(convParams.pads_begin);
    params["pads_end"] = propertyToString(convParams.pads_end);
    params["dilations"] = propertyToString(convParams.dilation);
    return params;
}

size_t ConvolutionTestHelper::getWeightByteSize(size_t elementSize, size_t numChannels) const {
    return (convParams.kernel[X_AXIS] * convParams.kernel[Y_AXIS] * convParams.out_c * numChannels * elementSize)
           / convParams.group;
}

size_t ConvolutionTestHelper::getBiasByteSize(size_t elementSize) const { return convParams.out_c * elementSize; }

void
ConvolutionTestHelper::ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst, const float *weights_data,
                                size_t weights_size, const float *bias_data, size_t bias_size) const {
    ref_conv_common<>(srcs, dst, weights_data, weights_size, bias_data, bias_size, convParams);
}

void ConvolutionTestHelper::ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                     const ie_fp16 *weights_data, size_t weights_size,
                                     const ie_fp16 *bias_data, size_t bias_size) const {
    ref_conv_common<>(srcs, dst, weights_data, weights_size, bias_data, bias_size, convParams);
}

DeconvolutionTestHelper::DeconvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams) : ConvolutionTestHelper(
        _convParams) {
    type = "Deconvolution";
}

void
DeconvolutionTestHelper::ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                  const float *weights_data,
                                  size_t weights_size, const float *bias_data, size_t bias_size) const {
    ref_deconv_common<float>(srcs, dst, weights_data, weights_size, bias_data, bias_size, convParams);
}

void DeconvolutionTestHelper::ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                       const ie_fp16 *weights_data, size_t weights_size,
                                       const ie_fp16 *bias_data, size_t bias_size) const {
    ref_deconv_common<ie_fp16>(srcs, dst, weights_data, weights_size, bias_data, bias_size, convParams);
}


DeformableConvolutionTestHelper::DeformableConvolutionTestHelper(const CommonTestUtils::conv_common_params &_convParams,
                                                                 const int deformable_group) :
                                                                 defConvParams(convParams), ConvolutionTestHelper( _convParams) {
    defConvParams.deformable_group = deformable_group;
    type = "DeformableConvolution";
}

void DeformableConvolutionTestHelper::ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                  const float *weights_data,
                                  size_t weights_size, const float *bias_data, size_t bias_size) const {
    ref_def_conv_common<float>(srcs, dst, weights_data, weights_size, bias_data, bias_size, defConvParams);
}

void DeformableConvolutionTestHelper::ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                       const ie_fp16 *weights_data, size_t weights_size,
                                       const ie_fp16 *bias_data, size_t bias_size) const {
    ref_def_conv_common<ie_fp16>(srcs, dst, weights_data, weights_size, bias_data, bias_size, defConvParams);
}

void DeformableConvolutionTestHelper::updatePaddingValues(const CNNNetwork &network) {
    details::CNNNetworkIterator i(network), end;
    auto found = std::find_if(i, end, [this](const CNNLayer::Ptr &layer) {
        return layer->type == type;
    });
    ASSERT_NE(found, end);

    auto castedLayer = std::dynamic_pointer_cast<ConvolutionLayer>(*found);
    auto allPad = getPaddings(*castedLayer.get());
    defConvParams.pads_end = allPad.end;
    defConvParams.pads_begin = allPad.begin;
}

std::map<std::string, std::string> DeformableConvolutionTestHelper::getMapParams() const {
    std::map<std::string, std::string> params;
    if (!defConvParams.auto_pad.empty()) {
        params["auto_pad"] = defConvParams.auto_pad;
    }
    params["group"] = std::to_string(defConvParams.group);
    params["output"] = std::to_string(defConvParams.out_c);
    params["deformable_group"] = std::to_string(defConvParams.deformable_group);

    auto propertyToString = [](const PropertyVector<unsigned int> &propertyVector) -> std::string {
        if (!propertyVector.size()) return "";
        std::string result = std::to_string(propertyVector[0]);
        for (int i = 1; i < propertyVector.size(); i++) {
            result += "," + std::to_string(propertyVector[i]);
        }
        return result;
    };
    params["kernel"] = propertyToString(defConvParams.kernel);
    params["strides"] = propertyToString(defConvParams.stride);
    params["pads_begin"] = propertyToString(defConvParams.pads_begin);
    params["pads_end"] = propertyToString(defConvParams.pads_end);
    params["dilations"] = propertyToString(defConvParams.dilation);
    return params;
}

PoolingTestHelper::PoolingTestHelper(const CommonTestUtils::pool_common_params &_poolParams) : LayerTestHelper("Pooling"),
                                                                              poolParams(_poolParams) {
}

std::map<std::string, std::string> PoolingTestHelper::getMapParams() const {
    std::map<std::string, std::string> params;
    if (!poolParams.auto_pad.empty()) {
        params["auto_pad"] = poolParams.auto_pad;
    }
    params["kernel"] = propertyToString(poolParams.kernel);
    params["strides"] = propertyToString(poolParams.stride);
    auto padStr = propertyToString(poolParams.pads_begin);
    if (!padStr.empty()) params["pads_begin"] = padStr;
    padStr = propertyToString(poolParams.pads_end);
    if (!padStr.empty()) params["pads_end"] = padStr;
    params["exclude-pad"] = poolParams.exclude_pad ? "true" : "false";
    params["pool-method"] = poolParams.avg ? "avg" : "max";
    return params;
}

void
PoolingTestHelper::ref_fp32(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                            const float *weights_data, size_t weights_size,
                            const float *bias_data, size_t bias_size) const {
    ref_pool_common<float>(srcs, dst, poolParams);
}

void PoolingTestHelper::ref_fp16(const std::vector<InferenceEngine::Blob::Ptr> srcs, Blob &dst,
                                 const ie_fp16 *weights_data, size_t weights_size,
                                 const ie_fp16 *bias_data, size_t bias_size) const {
    ref_pool_common<ie_fp16>(srcs, dst, poolParams);
}

void PoolingTestHelper::updatePaddingValues(const InferenceEngine::CNNNetwork &network) {
    details::CNNNetworkIterator i(network), end;
    auto found = std::find_if(i, end, [this](const CNNLayer::Ptr &layer) {
        return layer->type == type;
    });
    ASSERT_NE(found, end);

    auto castedLayer = std::dynamic_pointer_cast<PoolingLayer>(*found);
    auto allPad = getPaddings(*castedLayer.get());
    poolParams.pads_end = allPad.end;
    poolParams.pads_begin = allPad.begin;
}

size_t PoolingTestHelper::getWeightByteSize(size_t elementSize, size_t numChannels) const {
    return 0;
}

size_t PoolingTestHelper::getBiasByteSize(size_t elementSize) const {
    return 0;
}

class CommonSingleLayerTest
        : public testing::WithParamInterface<std::tuple<InitialShapes, NewShapes, PluginParams, Helper>>,
          public ::testing::Test {
protected:
    void SetUp() override {
        auto params = GetParam();
        initialShapes = std::get<0>(params);
        newShapes = std::get<1>(params);
        pluginParams = std::get<2>(params);
        layerHelper = std::get<3>(params);
        PluginCache::get().reset();
    }

    ICNNNetwork::InputShapes
    setInputShapes(CNNNetwork &network, const std::vector<SizeVector> &dims) {
        auto inputShapes = network.getInputShapes();
        int i = 0;
        IE_ASSERT(inputShapes.size() == dims.size());
        for (auto &pair : inputShapes) {
            pair.second = dims[i++];
        }
        return inputShapes;
    }

    TBlob<uint8_t>::Ptr createWeights(size_t elementSize, size_t weightByteSize, size_t biasByteSize) const {
        TBlob<uint8_t>::Ptr weights = make_shared_blob<uint8_t>({Precision::U8, {weightByteSize + biasByteSize}, Layout::C});
        weights->allocate();
        BufferWrapper wrappedWeights(weights, this->pluginParams.precision);
        fill_data_common(wrappedWeights, weights->size() / elementSize);
        return weights;
    }

    template<int Version = 3>
    static InferenceEngine::CNNNetwork
    buildSingleLayerNetwork(const std::string &layerType,
                            const CommonTestUtils::InOutShapes &inOutShapes,
                            std::map<std::string, std::string> *params,
                            const std::string &layerDataName = "data",
                            const Precision &precision = Precision::FP32,
                            size_t weightsSize = 0,
                            size_t biasesSize = 0,
                            const TBlob<uint8_t>::Ptr &weights = nullptr) {
        return buildSingleLayerNetworkCommon<Version>(layerType, inOutShapes, params, layerDataName, precision,
                                                      weightsSize, biasesSize, weights);
    }

protected:
    CommonTestUtils::InOutShapes initialShapes;
    CommonTestUtils::InOutShapes newShapes;
    PluginDependentParam pluginParams;
    LayerTestHelper::Ptr layerHelper;

    InputInfo::Ptr inputData;
    std::string inputName;
    InputInfo::Ptr transData;
    std::string transName;
    DataPtr outputData;
    std::string outputName;
};

TEST_P(CommonSingleLayerTest, inferAfterReshape) {
    Core ie;

    auto params = layerHelper->getMapParams();
    size_t elementSize = Precision(pluginParams.precision).size();
    ASSERT_EQ(initialShapes.inDims[0][1], newShapes.inDims[0][1]);
    size_t numChannels = initialShapes.inDims[0][1];
    size_t weightByteSize = layerHelper->getWeightByteSize(elementSize, numChannels);
    size_t biasByteSize = layerHelper->getBiasByteSize(elementSize);

    auto weights = createWeights(elementSize, weightByteSize, biasByteSize);

    auto network = buildSingleLayerNetwork<3>(layerHelper->getType(), initialShapes, &params, "data",
                                              pluginParams.precision, weightByteSize, biasByteSize, weights);

    std::tie(inputName, inputData) = (*network.getInputsInfo().begin());
    inputData->setPrecision(pluginParams.precision);
    inputData->setLayout(pluginParams.layout);
    std::tie(outputName, outputData) = (*network.getOutputsInfo().begin());
    outputData->setPrecision(pluginParams.precision);
    outputData->setLayout(pluginParams.layout);

    if (layerHelper->getType() == "DeformableConvolution") {
        std::tie(transName, transData) = (*network.getInputsInfo().find("Input1"));
        transData->setPrecision(pluginParams.precision);
        transData->setLayout(pluginParams.layout);
    }

    auto inputShapes = setInputShapes(network, newShapes.inDims);

    network.reshape(inputShapes);
    layerHelper->updatePaddingValues(network);

    auto exeNetwork = ie.LoadNetwork(network, pluginParams.deviceName);
    auto request = exeNetwork.CreateInferRequest();
    auto src = request.GetBlob(inputName);
    GenRandomDataCommon(src);

    size_t weights_size = weightByteSize / elementSize;
    size_t biases_size = biasByteSize / elementSize;

    if (layerHelper->getType() == "DeformableConvolution") {
        auto trans = request.GetBlob(transName);
        GenRandomDataCommon(trans);

        request.Infer();
        auto dst = request.GetBlob(outputName);

        Blob::Ptr dst_ref = layerHelper->getRefBlob(weights_size, biases_size, weights, { src, trans },
                                                    dst->getTensorDesc(), pluginParams.precision);
        CompareCommonAbsolute(dst, dst_ref, pluginParams.tolerance);

        BufferWrapper src_ptr(src);
        BufferWrapper trans_ptr(trans);
        BufferWrapper dst_ptr(dst_ref);
    } else {
        request.Infer();
        auto dst = request.GetBlob(outputName);

        Blob::Ptr dst_ref = layerHelper->getRefBlob(weights_size, biases_size, weights, { src },
                                                    dst->getTensorDesc(), pluginParams.precision);

        CompareCommonAbsolute(dst, dst_ref, pluginParams.tolerance);
    }
}
