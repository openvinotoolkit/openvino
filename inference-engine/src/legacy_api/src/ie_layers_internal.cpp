// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers_internal.hpp"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "layer_transform.hpp"

namespace InferenceEngine {

template <class Layer>
int getKernel(const Layer& layer, size_t i) {
    if (layer._dilation.size() > i && layer._dilation[i]) return (layer._kernel[i] - 1) * layer._dilation[i] + 1;
    return layer._kernel[i];
}

template <>
int getKernel(const PoolingLayer& layer, size_t i) {
    return layer._kernel[i];
}

template <class Layer>
Paddings getPaddingsInternal(const Layer& layer) {
    std::string errorPrefix = "Failed to calculate padding for " + layer.type + ": ";
    try {
        const std::map<std::string, std::string>& params = layer.params;
        const std::vector<DataWeakPtr>& insData = layer.insData;
        auto it = params.find("auto_pad");
        if (it != params.end()) {
            if (it->second == "valid") {
                return {PropertyVector<unsigned>(layer._kernel.size(), 0u),
                        PropertyVector<unsigned>(layer._kernel.size(), 0u)};
            } else {
                if ((insData.size() > 3 || insData.empty()) && layer.type != "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be in range [1, 3]";
                if ((insData.size() > 4 || insData.empty()) && layer.type == "DeformableConvolution")
                    THROW_IE_EXCEPTION << "number of inputs should be in range [2, 4]";
                auto firstInput = insData[0].lock();
                if (!firstInput) THROW_IE_EXCEPTION << "input is empty";
                auto shape = firstInput->getTensorDesc().getDims();
                auto shape_size = shape.size();
                if (shape_size < 4 || shape_size > 5) THROW_IE_EXCEPTION << "input shape must be 4D or 5D";

                std::vector<int> shapes;
                shapes.push_back(shape[shape_size - 1]);
                shapes.push_back(shape[shape_size - 2]);
                if (shape_size > 4) shapes.push_back(shape[shape_size - 3]);

                PropertyVector<unsigned int> pad_begin, pad_end;

                bool same_upper = it->second == "same_upper";
                bool same_lower = it->second == "same_lower";
                bool is_deconv = (layer.type == "Deconvolution");

                for (size_t i = 0; i < layer._kernel.size(); i++) {
                    int PA = 0;
                    int kernel = getKernel(layer, i);

                    int stride = layer._stride.size() > i ? layer._stride[i] : 1;
                    int sh = shapes[i];
                    if (is_deconv) sh *= stride;

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
    } catch (const InferenceEngine::details::InferenceEngineException& iee) {
        THROW_IE_EXCEPTION << errorPrefix << iee.what();
    }
}

class PaddingsUpdater {
    std::reference_wrapper<Paddings> pad;

public:
    explicit PaddingsUpdater(Paddings& pad): pad(pad) {}
    template <class T>
    typename std::enable_if<!std::is_same<T, CNNLayer*>::value, bool>::type operator()(T& layer) const {
        pad.get() = getPaddingsInternal(*layer);
        return true;
    }
    bool operator()(CNNLayer* layer) const {
        THROW_IE_EXCEPTION << "padding calculation for layer: " << layer->name << "(" << layer->type << ") unsupported";
    }
};

Paddings getPaddingsImpl(const CNNLayer& layer) {
    Paddings actual;
    details::visitActualLayer(std::tuple<DeformableConvolutionLayer*, DeconvolutionLayer*, ConvolutionLayer*,
                                         BinaryConvolutionLayer*, PoolingLayer*, CNNLayer*>(),
                              layer, PaddingsUpdater(actual));
    return actual;
}

int getNumIteration(const TensorIterator& tensorIterator) {
    using PortMap = TensorIterator::PortMap;
    const auto isIterable = [](const PortMap& rule) { return rule.axis != -1; };
    const auto getNumIterations = [](const PortMap& rule, const DataPtr& iterableData) -> int {
        if (iterableData == nullptr) {
            THROW_IE_EXCEPTION << ": Iteration over an invalid data object (null pointer dereference)";
        }
        const auto& dimensions = iterableData->getDims();

        const auto axis = rule.axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {
            THROW_IE_EXCEPTION << R"(: Invalid "axis" value in an iteration component: )"
                               << rule.axis  << ", dimensions number = " << dimensions.size() << " (out of range)";
        }
        const auto space = dimensions[axis];
        const int start = (rule.start < 0 ? (space + 1) : 0) + rule.start;
        const int end   = (rule.end   < 0 ? (space + 1) : 0) + rule.end;

        const auto stride = rule.stride;
        if (stride == 0) {
            THROW_IE_EXCEPTION << R"(: Invalid "stride" value in an iteration component: )" << rule.stride << " (infinite loop)";
        }
        const auto step = std::abs(stride);

        const auto src = stride < 0 ? end : start;
        const auto dst = stride < 0 ? start : end;
        const auto length = dst - src;
        if (src < 0 || src >= dst || dst > space || length < step) {
            THROW_IE_EXCEPTION << R"(: Invalid "start"/"stride"/"end" values in an iteration component)"
                               << ": \"start\" = " << rule.start << ", \"stride\" = " << rule.stride  << ", \"end\" = " << rule.end;
        }

        if (length % step != 0) {
            THROW_IE_EXCEPTION << ": Each iteration must be the same size: length (" << length << ") is not divisible by step (" << step << ")";
        }

        return static_cast<int>(length / step);
    };


    int numIterations = 1;
    bool isDefault = true;
    for (const auto& rule : tensorIterator.input_port_map) {
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= tensorIterator.insData.size()) {
            THROW_IE_EXCEPTION << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << tensorIterator.insData.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, tensorIterator.insData[rule.from].lock());
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_IE_EXCEPTION << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    for (const auto& rule : tensorIterator.output_port_map) {
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= tensorIterator.outData.size()) {
            THROW_IE_EXCEPTION << R"(: Invalid "from" value: "from" = )" << rule.from
                               << " inputs number = " << tensorIterator.outData.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, tensorIterator.outData[rule.from]);
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_IE_EXCEPTION << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    return numIterations;
}


}  // namespace InferenceEngine
