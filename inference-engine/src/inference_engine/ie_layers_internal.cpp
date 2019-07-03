// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "ie_layers_internal.hpp"
#include "layer_transform.hpp"

namespace InferenceEngine {

template<class Layer>
int getKernel(const Layer &layer, size_t i) {
    if (layer._dilation.size() > i && layer._dilation[i])
        return (layer._kernel[i] - 1) * layer._dilation[i] + 1;
    return layer._kernel[i];
}

template<>
int getKernel(const PoolingLayer &layer, size_t i) {
    return layer._kernel[i];
}

template<class Layer>
Paddings getPaddingsInternal(const Layer &layer) {
    std::string errorPrefix = "Failed to calculate padding for " + layer.type + ": ";
    try {
        const std::map<std::string, std::string> &params = layer.params;
        const std::vector<DataWeakPtr> &insData = layer.insData;
        auto it = params.find("auto_pad");
        if (it != params.end()) {
            if (it->second == "valid") {
                return {PropertyVector<unsigned>(layer._kernel.size(), 0u),
                        PropertyVector<unsigned>(layer._kernel.size(), 0u)};
            } else {
                if (insData.size() != 1)
                    THROW_IE_EXCEPTION << "number of inputs should be equal 1";
                auto firstInput = insData[0].lock();
                if (!firstInput)
                    THROW_IE_EXCEPTION << "input is empty";
                auto shape = firstInput->getTensorDesc().getDims();
                auto shape_size = shape.size();
                if (shape_size < 4 || shape_size > 5)
                    THROW_IE_EXCEPTION << "input shape must be 4D or 5D";

                std::vector<int> shapes;
                shapes.push_back(shape[shape_size - 1]);
                shapes.push_back(shape[shape_size - 2]);
                if (shape.size() > 4)
                    shapes.push_back(shape[shape_size - 3]);

                PropertyVector<unsigned int> pad_begin, pad_end;

                for (size_t i = 0; i < layer._kernel.size(); i++) {
                    int PA = 0;
                    int kernel = getKernel(layer, i);

                    int stride = layer._stride.size() > i ? layer._stride[i] : 1;
                    int sh = shapes[i];
                    if (sh % stride == 0) {
                        PA = std::max(kernel - stride, 0);
                    } else {
                        PA = std::max(kernel - (sh % stride), 0);
                    }
                    unsigned p_begin = PA / 2;
                    unsigned p_end = PA - p_begin;

                    pad_begin.insert(i, p_begin);
                    pad_end.insert(i, p_end);
                }

                return {pad_begin, pad_end};
            }
        }
        return {layer._padding, layer._pads_end};
    } catch (const InferenceEngine::details::InferenceEngineException &iee) {
        THROW_IE_EXCEPTION << errorPrefix << iee.what();
    }
}

class PaddingsUpdater {
    std::reference_wrapper<Paddings> pad;
 public:
    explicit PaddingsUpdater(Paddings & pad) : pad(pad) {}
    template <class T>
    typename std::enable_if<!std::is_same<T, CNNLayer*>::value, bool>::type
    operator () (T & layer) const {
        pad.get() = getPaddingsInternal(*layer);
        return true;
    }
    bool operator () (CNNLayer * layer) const {
        THROW_IE_EXCEPTION << "padding calculation for layer: " << layer->name << "(" << layer->type << ") unsupported";
    }
};

Paddings getPaddingsImpl(const CNNLayer &layer) {
    Paddings actual;
    details::visitActualLayer(std::tuple <DeconvolutionLayer*, ConvolutionLayer*, BinaryConvolutionLayer*, PoolingLayer*,
            CNNLayer*>(), layer, PaddingsUpdater(actual));
    return actual;
}

}  // namespace InferenceEngine
