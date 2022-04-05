// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <ie_blob.h>

#include "vpu_test_common_definitions.hpp"
#include "single_layer_common.hpp"
#include "myriad_layers_reference_functions.hpp"

class VpuTestNet
{
public:
    using CallbackBasic = std::function<void(InferenceEngine::Blob::Ptr inTensor,
                                             InferenceEngine::Blob::Ptr outTensor,
                                             const ParamsStruct& params)>;

    using CallbackWithWeights = std::function<void(const InferenceEngine::Blob::Ptr src,
                                                   InferenceEngine::Blob::Ptr dst,
                                                   const uint16_t *weights,
                                                   size_t weightsSize,
                                                   const uint16_t *biases,
                                                   size_t biasSize,
                                                   const ParamsStruct& params)>;


    using CalcWeights = std::function<void(uint16_t* ptr, size_t weightsSize)>;

    class LayerParams {
    public:
        std::string _layerType;
        std::string _layerName;
        ParamsStruct _params;
        size_t _weightsSize = 0;
        size_t _biasesSize = 0;
        CalcWeights _fillWeights;
        CalcWeights _fillBiases ;
        IN_OUT_desc _inDim;
        IN_OUT_desc _outDim;
        IN_OUT_desc _weightsDim;
        IN_OUT_desc _biasesDim;
        InferenceEngine::Precision _outPrecision = InferenceEngine::Precision::FP16;
    };

    class LayerInitParams : public LayerParams {
    public:
        LayerInitParams(const std::string& layerType) { _layerType = layerType; _layerName = layerType + "_TEST"; }

        LayerInitParams& name(const std::string& name)
            { _layerName = name; return *this;}

        LayerInitParams& params(ParamsStruct params)
            { _params = std::move(params); return *this;}
        LayerInitParams& weights(const size_t weightsSize)
            { _weightsSize = weightsSize; return *this;}
        LayerInitParams& biases(const size_t biasesSize)
            { _biasesSize = biasesSize; return *this;}

        LayerInitParams& in(IN_OUT_desc inDim)
            { _inDim = std::move(inDim); return *this;}
        LayerInitParams& out(IN_OUT_desc outDim)
            { _outDim = std::move(outDim); return *this;}
        LayerInitParams& weightsDim(IN_OUT_desc weightsDim)
            { _weightsDim = std::move(weightsDim); return *this;}
        LayerInitParams& biasesDim(IN_OUT_desc biasesDim)
            { _biasesDim = std::move(biasesDim); return *this;}

        LayerInitParams& fillWeights(CalcWeights && fillWeightsCallback)
            { _fillWeights = std::move(fillWeightsCallback); return *this;}
        LayerInitParams& fillBiases(CalcWeights && fillBiasesCallback)
            { _fillBiases = std::move(fillBiasesCallback); return *this;}

        LayerInitParams& outPrecision(const InferenceEngine::Precision outPrecision)
            { _outPrecision = outPrecision; return *this;}
    };

    /* This is limited implementation of functionality required for graphs generation.  */
    /* The code allows to build linear chains of layers to provide testing of functions */
    /* with one input and one output                                                    */
    void addLayer(const LayerParams& params);
    void addLayer(const LayerParams& params, CallbackBasic&& callback);
    void addLayer(const LayerParams& params, CallbackWithWeights&& callback);

    void run() const;

    void clear();
    bool empty() const {
        return _layers.empty() && _callbacks.empty();
    }

    struct NetworkSerializedData {
        std::string model;
        WeightsBlob::Ptr weights;
    };

    NetworkSerializedData genNetwork(IRVersion version);
    void setWeightsCallbackForLayer(size_t index, CalcWeights&& callback);
    void setBiasesCallbackForLayer(size_t index, CalcWeights&& callback);

    InferenceEngine::Blob::Ptr getFirstInput() const;
    InferenceEngine::Blob::Ptr getLastOutput() const;

private:
    class ReferenceFunctionWrapper {
    public:
        std::function<void()> _callback;
        InferenceEngine::Blob::Ptr _input;
        InferenceEngine::Blob::Ptr _output;
        WeightsBlob::Ptr _weightsPtr;
        WeightsBlob::Ptr _biasesPtr;
        uint16_t* _weights = nullptr;
        uint16_t* _biases = nullptr;
        size_t _weightsSize = 0;
        size_t _biasesSize = 0;

    public:
        void setCallback(CallbackBasic&& f, const ParamsStruct& params);
        void setCallback(CallbackWithWeights&& f, const ParamsStruct& params);
    };

private:
    void genInputOutput(ReferenceFunctionWrapper& obj,
                        const LayerParams& params);

    ReferenceFunctionWrapper& addLayerImpl(const LayerParams& params);

private:
    std::vector<LayerParams> _layers;
    std::vector<ReferenceFunctionWrapper> _callbacks;
};

