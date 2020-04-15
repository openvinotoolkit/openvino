// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bf16transformer.h"
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <set>
#include <chrono>
#include "details/ie_cnn_network_tools.h"
#include "ie_util_internal.hpp"
#include "ngraph/type/bfloat16.hpp"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

void precisionColoringBF16(const CNNLayerPtr layer,
                           ordered_properties &printed_properties,
                           ordered_properties &node_properties) {
    if (layer && !layer->insData.empty() && layer->input()) {
        printed_properties.insert(printed_properties.begin(),
                                  std::pair<std::string, std::string>("Precision",
                                                                      layer->input()->getPrecision() == Precision::FP32 ? "FP32" : "BF16"));

        if (layer->input()->getPrecision() == Precision::FP32) {
            node_properties.emplace_back("fillcolor", "#5A5DF0");
        } else {
            node_properties.emplace_back("fillcolor", "#20F608");
        }
    }
}

void BF16Transformer::convertToFloat(InferenceEngine::CNNNetwork &network) {
    // go over all edges and all edges having FP32 mark as BF16
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();
    for (auto iter : sortedLayers) {
        for (size_t o = 0; o < iter->outData.size(); o++) {
            if (inputs.find(iter->outData[o]->getName()) == inputs.end()
                && outputs.find(iter->outData[o]->getName()) == outputs.end()
                && iter->outData[o]->getPrecision() == Precision::BF16) {
                iter->outData[o]->setPrecision(Precision::FP32);
            }
        }
    }
}

void BF16Transformer::convertToBFloat16(InferenceEngine::CNNNetwork &network) {
    // go over all edges and all edges having FP32 mark as BF16
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    InputsDataMap inputs = network.getInputsInfo();
    OutputsDataMap outputs = network.getOutputsInfo();
    for (auto iter : sortedLayers) {
        for (size_t o = 0; o < iter->outData.size(); o++) {
            if (inputs.find(iter->outData[o]->getName()) == inputs.end()
                && outputs.find(iter->outData[o]->getName()) == outputs.end()
                && iter->outData[o]->getPrecision() == Precision::FP32) {
                iter->outData[o]->setPrecision(Precision::BF16);
            }
        }
    }

    // convert all edges back to FP32 on demand
    optimizeToFloat(network);
}

void BF16Transformer::optimizeToFloat(InferenceEngine::CNNNetwork &network) {
    std::set<DataPtr> toAnalyzeTensors;
    std::set<DataPtr> immutable;
    bool hasBF16Tensor = false;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    // 1. Verify if we do not have bf16 tensors - it's better to return early and not to try to return anything since there is no such tensors
    for (auto iter : sortedLayers) {
        for (size_t i = 0; i < iter->insData.size(); i++) {
            if (iter->insData[i].lock()->getTensorDesc().getPrecision() == Precision::BF16) {
                hasBF16Tensor = true;
            }
        }
        for (size_t o = 0; o < iter->outData.size(); o++) {
            if (iter->outData[o]->getTensorDesc().getPrecision() == Precision::BF16) {
                hasBF16Tensor = true;
            }
        }
    }
    if (!hasBF16Tensor) {
        return;
    }
    // 2a. go over all inputs and outputs and put them to the toAnalyzeTensors
    InputsDataMap inputs = network.getInputsInfo();
    for (auto input : inputs) {
        immutable.insert(input.second->getInputData());
        if (input.second->getInputData()->getTensorDesc().getPrecision() != Precision::BF16) {
            toAnalyzeTensors.insert(input.second->getInputData());
        }
    }

    OutputsDataMap outputs = network.getOutputsInfo();
    for (auto output : outputs) {
        immutable.insert(output.second);
        if (output.second->getTensorDesc().getPrecision() != Precision::BF16) {
            toAnalyzeTensors.insert(output.second);
        }
    }

    // 2b. go over all unknown layers for this algo and mark them as fp32 and add to the toAnalyzeTensors
    // 2c. go over all inputs to _initbf16 and if they are fp32 - add them to the toAnalyzeTensors
    for (auto iter : sortedLayers) {
        if (_initbf16.find(iter->type) == _initbf16.end()
            && _complementbf16.find(iter->type) == _complementbf16.end()
            && _multiinput.find(iter->type) == _multiinput.end()) {
            // try to mark inputs of the unknown layer
            for (size_t i = 0; i < iter->insData.size(); i++) {
                if (iter->insData[i].lock()->getPrecision() == Precision::BF16) {
                    bool marked = tryToMarkFP32(iter->insData[i].lock(), immutable);
                    if (marked) {
                        toAnalyzeTensors.insert(iter->insData[i].lock());
                    }
                }
            }
            // try to mark outputs of the unknown layer
            for (size_t o = 0; o < iter->outData.size(); o++) {
                if (iter->outData[o]->getPrecision() == Precision::BF16) {
                    bool marked = tryToMarkFP32(iter->outData[o], immutable);
                    if (marked) {
                        toAnalyzeTensors.insert(iter->outData[o]);
                    }
                }
            }
        }
        if (_initbf16.find(iter->type) != _initbf16.end()) {
            // verify if input activation tensor is not bf16 - add to toAnalyzeTensors as well
            // we are assuming here that _initbf16 contain only layers having one dynamic input
            // in other case algorithm should be changed to care about two dynamic input tensors
            // and take into account case of different precision if they are
            if (iter->insData[0].lock()->getTensorDesc().getPrecision() != Precision::BF16) {
                toAnalyzeTensors.insert(iter->insData[0].lock());
                // output tensor for FP32 convolutoin/FC layers should be FP32 as well
                for (size_t o = 0; o < iter->outData.size(); o++) {
                    if (iter->outData[o]->getPrecision() == Precision::BF16) {
                        bool marked = tryToMarkFP32(iter->outData[o], immutable);
                        if (marked) {
                            toAnalyzeTensors.insert(iter->outData[o]);
                        }
                    }
                }
            }
        }
    }

    // 3 - while toAnalyzeTensors is not empty look at the layers dealing with tensors mentioned in toAnalyzeTensors
    while (!toAnalyzeTensors.empty()) {
        DataPtr tensor = *toAnalyzeTensors.begin();
        toAnalyzeTensors.erase(tensor);
        // look into producer of the tensor
        auto layer = tensor->getCreatorLayer().lock();
        // if this layer is not from _initbf16 - analyze inputs
        if (_initbf16.find(layer->type) == _initbf16.end()) {
            // for all inputs investigate and modify tensor precision if required
            for (size_t i = 0; i < layer->insData.size(); i++) {
                bool marked = tryToMarkFP32(layer->insData[i].lock(), immutable);
                if (marked) {
                    toAnalyzeTensors.insert(layer->insData[i].lock());
                }
            }
        }

        // mark all produced tensors to FP32 if they are BF16 and if they do not go _only_ to the toAnalyzeTensors
        // TODO: when we enable greedy mode and start to produce bf16 tensor even if one consumer accepts it,
        // this place should be changed.
        // Instead of "if they do not go _only_ to the toAnalyzeTensors" we have to apply "if they do not go at least to one of _initbf16"
        // TODO: add test input1->pooling1->conv1 and the same pooling1->relu. for example. now convolution should be returned to fp32
        // after greedy mode, it should be fp32.
        for (auto inputTo : tensor->getInputTo()) {
            for (size_t o = 0; o < inputTo.second->outData.size(); o++) {
                if (inputTo.second->outData[o]->getTensorDesc().getPrecision() == Precision::BF16) {
                    bool marked = tryToMarkFP32(inputTo.second->outData[o], immutable);
                    if (marked) {
                        toAnalyzeTensors.insert(layer->outData[o]);
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::ofstream file("bf16_icnnnetwork.dot");
        saveGraphToDot(network, file, precisionColoringBF16);
    }
#endif
}

bool BF16Transformer::tryToMarkFP32(InferenceEngine::DataPtr data, const std::set<InferenceEngine::DataPtr>& immutable) {
    bool marked = false;
    if (immutable.find(data) == immutable.end() && data->getPrecision() == Precision::BF16) {
        // we treat one consumer and many in different ways
        // if there is one consumer, we can mark its input as float if it does not belong to the list of initial layers
        // in other cases we need to mark tensor which is passed to several l ayers as FP32 only if there is at least one conusmer
        // produces data in FP32. I.e. there should be a way fo getting FP32 from output data to this point
        if (data->getInputTo().size() == 1) {
            if (_initbf16.find(data->getInputTo().begin()->second->type) == _initbf16.end()) {
                marked = true;
            }
        } else {
            // get all consumers
            for (auto o : data->getInputTo()) {
                // if tensor goes to several layers, we will mark it by FP32 only if one of the layer is unknown
                if (_initbf16.find(o.second->type) == _initbf16.end() &&
                    _complementbf16.find(o.second->type) == _complementbf16.end() &&
                    _multiinput.find(o.second->type) == _multiinput.end()) {
                    marked = true;
                }
            }
        }
        if (marked) {
            data->setPrecision(Precision::FP32);
        }
    }
    return marked;
}

InferenceEngine::MemoryBlob::Ptr BF16Transformer::convertBF16ToFloat(InferenceEngine::MemoryBlob::Ptr tweights) {
    TensorDesc td(Precision::FP32, tweights->getTensorDesc().getDims(), tweights->getTensorDesc().getLayout());
    MemoryBlob::Ptr weightsFP32 = make_shared_blob<float>(td);
    weightsFP32->allocate();
    auto lmbf16 = tweights->rmap();
    short *bf16data = lmbf16.as<short *>();
    auto lmfp32 = weightsFP32->wmap();
    float *fp32data = lmfp32.as<float *>();
    for (size_t i = 0; i < weightsFP32->size(); i++) {
        fp32data[i] = ngraph::bfloat16::from_bits(bf16data[i]);
    }
    return weightsFP32;
}
