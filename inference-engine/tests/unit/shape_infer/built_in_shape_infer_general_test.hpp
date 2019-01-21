// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <inference_engine/shape_infer/built-in/ie_built_in_holder.hpp>
#include <utility>
#include <inference_engine/ie_format_parser.h>
#include <xml_helper.hpp>
#include <xml_net_builder.hpp>
#include <single_layer_common.hpp>

class BaseTestCreator {
protected:
    std::string _type;
public:
    explicit BaseTestCreator(const std::string &type) : _type(type) {}

    virtual InferenceEngine::CNNLayerPtr create(const std::string &type) = 0;

    virtual bool shouldCreate(const std::string &type) = 0;
};

template<class LT>
class LayerTestCreator : public BaseTestCreator {
public:
    explicit LayerTestCreator(const std::string &type) : BaseTestCreator(type) {}

    InferenceEngine::CNNLayerPtr create(const std::string &type) override {
        InferenceEngine::LayerParams params;
        params.type = type;
        return std::make_shared<LT>(params);
    }

    bool shouldCreate(const std::string &type) override {
        return type == _type;
    }
};

struct param_size {
    unsigned x;
    unsigned y;

    friend std::ostream &operator<<(std::ostream &os, param_size const &paramSize) {
        os << "x=" << std::to_string(paramSize.x) << ", y=" << std::to_string(paramSize.y);
        return os;
    };

    std::string toSeparetedRow(const char *separator) {
        std::string res = std::to_string(y) + separator + std::to_string(x);
        return res;
    }
};

PRETTY_PARAM(kernel, param_size);

PRETTY_PARAM(stride, param_size);

PRETTY_PARAM(pad, param_size);

PRETTY_PARAM(padrb, param_size);

PRETTY_PARAM(auto_pad, std::string);

PRETTY_PARAM(out_channels, unsigned);

PRETTY_PARAM(group, unsigned);

PRETTY_PARAM(dilation_factor, param_size);

PRETTY_PARAM(pool_type, std::string);

PRETTY_PARAM(exclude_pad, bool);

PRETTY_PARAM(LayerType, std::string)

PRETTY_PARAM(LayerDataName, std::string)

PRETTY_PARAM(InOutShapes, testing::InOutData)

PRETTY_PARAM(NewInOutShapes, testing::InOutData)

PRETTY_PARAM(MapParams, MapStrStr)

PRETTY_PARAM(CanInfer, bool);

PRETTY_PARAM(IsTransposed, bool);

PRETTY_PARAM(TopologyPath, std::string);

PRETTY_PARAM(ModelPath, std::string);

static size_t BATCH = 100;

class BuiltInShapeInferCommon : public ::testing::Test {
protected:
    void SetUp() override {
        holder = std::make_shared<InferenceEngine::ShapeInfer::BuiltInShapeInferHolder>();
    }

    InferenceEngine::IShapeInferImpl::Ptr getShapeInferImpl(const std::string &type) {
        InferenceEngine::IShapeInferImpl::Ptr impl;
        sts = holder->getShapeInferImpl(impl, type.c_str(), &resp);
        if (sts != InferenceEngine::StatusCode::OK) THROW_IE_EXCEPTION << resp.msg;
        return impl;
    }

protected:
    InferenceEngine::StatusCode sts = InferenceEngine::StatusCode::GENERAL_ERROR;
    InferenceEngine::ResponseDesc resp;
    std::shared_ptr<InferenceEngine::IShapeInferExtension> holder;
};

template<class T>
class BuiltInShapeInferTestWithParam : public BuiltInShapeInferCommon,
                                       public testing::WithParamInterface<T> {
    const std::vector<std::shared_ptr<BaseTestCreator>> &getCreators() const {
        // there should be unique_ptr but it cant be used with initializer lists
        static std::vector<std::shared_ptr<BaseTestCreator> > creators = {
                std::make_shared<LayerTestCreator<InferenceEngine::PowerLayer>>("Power"),
                std::make_shared<LayerTestCreator<InferenceEngine::ConvolutionLayer>>("Convolution"),
                std::make_shared<LayerTestCreator<InferenceEngine::DeconvolutionLayer>>("Deconvolution"),
                std::make_shared<LayerTestCreator<InferenceEngine::PoolingLayer>>("Pooling"),
                std::make_shared<LayerTestCreator<InferenceEngine::FullyConnectedLayer>>("InnerProduct"),
                std::make_shared<LayerTestCreator<InferenceEngine::FullyConnectedLayer>>("FullyConnected"),
                std::make_shared<LayerTestCreator<InferenceEngine::NormLayer>>("LRN"),
                std::make_shared<LayerTestCreator<InferenceEngine::NormLayer>>("Norm"),
                std::make_shared<LayerTestCreator<InferenceEngine::SoftMaxLayer>>("Softmax"),
                std::make_shared<LayerTestCreator<InferenceEngine::SoftMaxLayer>>("SoftMax"),
                std::make_shared<LayerTestCreator<InferenceEngine::GRNLayer>>("GRN"),
                std::make_shared<LayerTestCreator<InferenceEngine::MVNLayer>>("MVN"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReLULayer>>("ReLU"),
                std::make_shared<LayerTestCreator<InferenceEngine::ClampLayer>>("Clamp"),
                std::make_shared<LayerTestCreator<InferenceEngine::SplitLayer>>("Split"),
                std::make_shared<LayerTestCreator<InferenceEngine::SplitLayer>>("Slice"),
                std::make_shared<LayerTestCreator<InferenceEngine::ConcatLayer>>("Concat"),
                std::make_shared<LayerTestCreator<InferenceEngine::EltwiseLayer>>("Eltwise"),
                std::make_shared<LayerTestCreator<InferenceEngine::ScaleShiftLayer>>("ScaleShift"),
                std::make_shared<LayerTestCreator<InferenceEngine::PReLULayer>>("PReLU"),
                std::make_shared<LayerTestCreator<InferenceEngine::CropLayer>>("Crop"),
                std::make_shared<LayerTestCreator<InferenceEngine::ReshapeLayer>>("Reshape"),
                std::make_shared<LayerTestCreator<InferenceEngine::TileLayer>>("Tile"),
                std::make_shared<LayerTestCreator<InferenceEngine::BatchNormalizationLayer>>("BatchNormalization"),
                std::make_shared<LayerTestCreator<InferenceEngine::GemmLayer>>("Gemm"),
                std::make_shared<LayerTestCreator<InferenceEngine::PadLayer>>("Pad"),
                std::make_shared<LayerTestCreator<InferenceEngine::GatherLayer>>("Gather")
        };
        return creators;
    }

protected:
    InferenceEngine::DataPtr
    getNotEmptyData(std::string const &name = "", const InferenceEngine::SizeVector &dims = {}) {
        InferenceEngine::TensorDesc desc(InferenceEngine::Precision::UNSPECIFIED, dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        return std::make_shared<InferenceEngine::Data>(name, desc);
    }

    InferenceEngine::CNNLayer::Ptr createLayer(const std::string &type) const {
        for (auto &creator : getCreators()) {
            if (!creator->shouldCreate(type))
                continue;
            return creator->create(type);
        }
        static LayerTestCreator<InferenceEngine::GenericLayer> genericCreator("");
        return genericCreator.create(type);
    }

    void initLayer(const InferenceEngine::CNNLayerPtr &layer, const testing::InOutData &inOutData) {
        for (const auto &in:inOutData.inDims) {
            auto data = getNotEmptyData("", in);
            _savedData.push_back(data);
            layer->insData.push_back(data);
        }
        for (const auto &out:inOutData.outDims) {
            layer->outData.push_back(getNotEmptyData("", out));
        }
    }

    static testing::InOutData getFakeData(const testing::InOutData &inOutShapes) {
        testing::InOutData initial = inOutShapes;
        for (auto &dims : initial.inDims) {
            std::fill(dims.begin(), dims.end(), 1);
        }
        for (auto &dims : initial.outDims) {
            std::fill(dims.begin(), dims.end(), 1);
        }
        return initial;
    }

    static InferenceEngine::ICNNNetwork::InputShapes
    setInputShapes(const InferenceEngine::ICNNNetwork &cnnNetwork,
                   const std::vector<InferenceEngine::SizeVector> &shapesToSet) {
        InferenceEngine::ICNNNetwork::InputShapes inputShapes;
        InferenceEngine::InputsDataMap inputs;
        cnnNetwork.getInputsInfo(inputs);
        for (const auto &pair : inputs) {
            auto info = pair.second;
            if (info) {
                auto data = info->getInputData();
                if (data) {
                    inputShapes[data->name] = data->getTensorDesc().getDims();
                }
            }
        }
        int i = 0;
        for (auto &pair : inputShapes) {
            pair.second = shapesToSet[i++];
        }
        return inputShapes;
    }

    static void checkNetworkInOut(const InferenceEngine::ICNNNetwork &network,
                                  const testing::InOutData &inOutData) {
        InferenceEngine::InputsDataMap inputsDataMap;
        InferenceEngine::OutputsDataMap outputsDataMap;
        network.getInputsInfo(inputsDataMap);
        network.getOutputsInfo(outputsDataMap);
        int i = 0;
        for (auto pair : inputsDataMap) {
            ASSERT_EQ(inOutData.inDims[i++], pair.second->getTensorDesc().getDims());
        }
        i = 0;
        for (auto pair : outputsDataMap) {
            ASSERT_EQ(inOutData.outDims[i++], pair.second->getDims());
        }
    }

    template<int Version = 3>
    static InferenceEngine::details::CNNNetworkImplPtr
    buildSingleLayerNetwork(const std::string &layerType,
                            const testing::InOutData &inOutShapes,
                            std::map<std::string, std::string> *params,
                            const std::string &layerDataName = "data") {
        auto *parser = new InferenceEngine::details::FormatParser(Version);
        return buildSingleLayerNetworkCommon<Version>(parser, layerType, inOutShapes, params, layerDataName);
    }

protected:
    std::vector<InferenceEngine::SizeVector> outShapes;
    std::map<std::string, std::string> params;
    std::map<std::string, InferenceEngine::Blob::Ptr> blobs;
    std::vector<InferenceEngine::DataPtr> _savedData;
};

class BuiltInShapeInferImplTest
        : public BuiltInShapeInferTestWithParam<std::tuple<LayerType, InOutShapes, NewInOutShapes, MapParams, LayerDataName, CanInfer>> {
protected:
    void SetUp() override {
        BuiltInShapeInferCommon::SetUp();
        auto params = GetParam();
        type = std::get<0>(params);
        inOutShapes = std::get<1>(params);
        newInOutShapes = std::get<2>(params);
        layerParams = std::get<3>(params);
        layerDataName = std::get<4>(params);
        canInfer = std::get<5>(params);
    }

protected:
    std::string type;
    testing::InOutData inOutShapes;
    testing::InOutData newInOutShapes;
    MapStrStr layerParams;
    std::string layerDataName;
    bool canInfer{};
};

