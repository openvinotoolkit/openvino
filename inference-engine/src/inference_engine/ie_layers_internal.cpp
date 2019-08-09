// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include <set>
#include <math.h>
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
                if (insData.size() != 1 && layer.type != "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be equal 1";
                if (insData.size() != 2 && layer.type == "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be equal 2";
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
                if (shape_size > 4)
                    shapes.push_back(shape[shape_size - 3]);

                PropertyVector<unsigned int> pad_begin, pad_end;

                bool same_upper = it->second == "same_upper";
                bool same_lower = it->second == "same_lower";
                bool is_deconv = (layer.type == "Deconvolution");

                for (size_t i = 0; i < layer._kernel.size(); i++) {
                    float PA = 0;
                    int kernel = getKernel(layer, i);

                    int stride = layer._stride.size() > i ? layer._stride[i] : 1;
                    int sh = shapes[i];
                    if (is_deconv)
                        sh *= stride;

                    int rm = sh % stride;
                    if (rm == 0) {
                        PA = std::max(kernel - stride, 0);
                    } else {
                        PA = std::max(kernel - rm, 0);
                    }
                    float p_begin = PA * 0.5f, p_end = PA - p_begin;

                    if (same_upper) {
                        p_begin = std::floor(p_begin);
                        p_end = std::ceil(p_end);
                    } else if (same_lower) {
                        p_begin = std::ceil(p_begin);
                        p_end = std::floor(p_end);
                    }
                    pad_begin.insert(i, static_cast<unsigned int>(p_begin));
                    pad_end.insert(i, static_cast<unsigned int>(p_end));
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
    details::visitActualLayer(std::tuple <DeformableConvolutionLayer*, DeconvolutionLayer*, ConvolutionLayer*, BinaryConvolutionLayer*, PoolingLayer*,
            CNNLayer*>(), layer, PaddingsUpdater(actual));
    return actual;
}

int getNumIteration(const TensorIterator &ti) {
    int iter_num = 1;  // 1 means no iteration

    for (auto & rule : ti.input_port_map) {
        if (rule.axis == -1) continue;

        auto data = ti.insData[rule.from].lock();
        IE_ASSERT(data);

        auto shape = data->getDims();
        size_t size = shape[rule.axis];
        size_t step = std::abs(rule.stride);
        size_t cur_iter_size = size / step;

        if (iter_num == 1) {
            iter_num = cur_iter_size;
        } else {
            if (iter_num != cur_iter_size)
                return -1;  // TI is inconsistent
        }
    }

    for (auto & rule : ti.output_port_map) {
        if (rule.axis == -1) continue;

        auto data = ti.outData[rule.from];
        auto shape = data->getDims();

        size_t size = shape[rule.axis];
        size_t step = std::abs(rule.stride);
        size_t cur_iter_size = size / step;

        if (iter_num == 1) {
            iter_num = cur_iter_size;
        } else {
            if (iter_num != cur_iter_size)
                return -1;  // TI is inconsistent
        }
    }
    return iter_num;
}

}  // namespace InferenceEngine
