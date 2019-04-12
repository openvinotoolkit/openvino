// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <inference_engine/blob_factory.hpp>
#include <inference_engine/shape_infer/built-in/ie_built_in_holder.hpp>
#include <utility>
#include <inference_engine/ie_format_parser.h>
#include <xml_helper.hpp>
#include <xml_net_builder.hpp>
#include <single_layer_common.hpp>
#include <tests_common.hpp>

namespace IE = InferenceEngine;

struct param_size {
    // dimensions order: x, y, z, ...
    std::vector<unsigned> dims;
    param_size() {}
//    param_size(const std::vector<unsigned>& dims) {
//        this->dims = dims;
//    }
    param_size(std::initializer_list<unsigned> dims) {
        this->dims = dims;
    }
    bool empty() {
        return dims.empty();
    }

    friend std::ostream &operator<<(std::ostream &os, param_size const &paramSize) {
        auto d_size = paramSize.dims.size();
        if (d_size > 0) {
            os << "dims[" << std::to_string(0) << "]=" << std::to_string(paramSize.dims[0]);
            for (int i = 1; i < paramSize.dims.size(); i++)
                os << ", dims[" << std::to_string(i) << "]=" << std::to_string(paramSize.dims[i]);
        }
        return os;
    };

    std::string toSeparetedRow(const char *separator) {
        auto d_size = dims.size();
        std::string res;
        if (d_size > 0) {
            res = std::to_string(dims[d_size - 1]);
            for (int i = d_size - 2; i >= 0; i--) {
                res += separator + std::to_string(dims[i]);
            }
        }
        return res;
    }
};

PRETTY_PARAM(kernel, param_size);

PRETTY_PARAM(stride, param_size);

PRETTY_PARAM(pad, param_size);

PRETTY_PARAM(pad_end, param_size);

PRETTY_PARAM(auto_pad, std::string);

PRETTY_PARAM(out_channels, unsigned);

PRETTY_PARAM(group, unsigned);

PRETTY_PARAM(dilation_factor, param_size);

PRETTY_PARAM(pool_type, std::string);

PRETTY_PARAM(exclude_pad, bool);

PRETTY_PARAM(LayerType, std::string)

PRETTY_PARAM(LayerDataName, std::string)

PRETTY_PARAM(InOutShapes, testing::InOutShapes)

PRETTY_PARAM(NewInOutShapes, testing::InOutShapes)

PRETTY_PARAM(MapParams, MapStrStr)

PRETTY_PARAM(CanInfer, bool);

PRETTY_PARAM(IsTransposed, bool);

PRETTY_PARAM(TopologyPath, std::string);

PRETTY_PARAM(ModelPath, std::string);

static size_t BATCH = 100;

class BuiltInShapeInferCommon : public TestsCommon {
protected:
    void SetUp() override {
        holder = std::make_shared<IE::ShapeInfer::BuiltInShapeInferHolder>();
    }

    IE::IShapeInferImpl::Ptr getShapeInferImpl(const std::string &type) {
        IE::IShapeInferImpl::Ptr impl;
        sts = holder->getShapeInferImpl(impl, type.c_str(), &resp);
        if (sts != IE::StatusCode::OK) THROW_IE_EXCEPTION << resp.msg;
        return impl;
    }

protected:
    IE::StatusCode sts = IE::StatusCode::GENERAL_ERROR;
    IE::ResponseDesc resp;
    std::shared_ptr<IE::IShapeInferExtension> holder;
};

template<class T>
class BuiltInShapeInferTestWithParam : public BuiltInShapeInferCommon,
                                       public testing::WithParamInterface<T> {

protected:
    static std::vector<IE::Blob::CPtr> getBlobs(const std::vector<IE::SizeVector>& shapes) {
        std::vector<IE::Blob::CPtr> inBlobs;
        for (auto const& dims : shapes) {
            IE::TensorDesc desc(IE::Precision::FP32, dims, IE::TensorDesc::getLayoutByDims(dims));
            auto blob = make_blob_with_precision(desc);
            inBlobs.push_back(blob);
        }
        return inBlobs;
    }

    static IE::ICNNNetwork::InputShapes
    setInputShapes(const IE::ICNNNetwork &cnnNetwork,
                   const std::vector<IE::SizeVector> &shapesToSet) {
        IE::ICNNNetwork::InputShapes inputShapes;
        IE::InputsDataMap inputs;
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

    static void checkNetworkInOut(const IE::ICNNNetwork &network,
                                  const testing::InOutShapes &inOutData) {
        IE::InputsDataMap inputsDataMap;
        IE::OutputsDataMap outputsDataMap;
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
    static IE::details::CNNNetworkImplPtr
    buildSingleLayerNetwork(const std::string &layerType,
                            const testing::InOutShapes &inOutShapes,
                            std::map<std::string, std::string> *params,
                            const std::string &layerDataName = "data") {
        auto *parser = new IE::details::FormatParser(Version);
        return buildSingleLayerNetworkCommon<Version>(parser, layerType, inOutShapes, params, layerDataName);
    }

protected:
    std::vector<IE::SizeVector> outShapes;
    std::map<std::string, std::string> params;
    std::map<std::string, IE::Blob::Ptr> blobs;
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
    testing::InOutShapes inOutShapes;
    testing::InOutShapes newInOutShapes;
    MapStrStr layerParams;
    std::string layerDataName;
    bool canInfer{};
};

