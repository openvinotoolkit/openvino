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
#include <legacy/details/ie_cnn_network_tools.h>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_tools.hpp>
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
        if (CaselessEq<std::string>()(iter->type, "convolution")) {
            auto dims = iter->insData[0].lock()->getDims();
            if ((dims.size() == 4 || dims.size() == 5) && (dims[1] == 1 || dims[1] == 3))
                continue;
        }

        //  check, if memory output node needs to be transformed
        if (iter->type == "Memory" && iter->outData.size() == 0 &&
            iter->insData[0].lock()->getPrecision() == Precision::FP32) {
            iter->insData[0].lock()->setPrecision(Precision::BF16);
        }

        for (size_t o = 0; o < iter->outData.size(); o++) {
            if (inputs.find(iter->outData[o]->getName()) == inputs.end()
                && outputs.find(iter->outData[o]->getName()) == outputs.end()
                && !CaselessEq<std::string>()(iter->type, "const")
                && iter->outData[o]->getPrecision() == Precision::FP32) {
                iter->outData[o]->setPrecision(Precision::BF16);
            }
        }
    }

    // insert convert after input if necessary
    insertConvertAfterInput(network);

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
        auto layer = getCreatorLayer(tensor).lock();
        // if this layer is not from _initbf16 - analyze inputs
        if (_initbf16.find(layer->type) == _initbf16.end()) {
            // for all inputs investigate and modify tensor precision if required
            for (size_t i = 0; i < layer->insData.size(); i++) {
                auto creator = getCreatorLayer(layer->insData[i].lock());
                if (_skipmarking.find(creator.lock()->type) != _skipmarking.end()) {
                    continue;
                }
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
        for (auto inputTo : getInputTo(tensor)) {
            for (size_t o = 0; o < inputTo.second->outData.size(); o++) {
                if (inputTo.second->outData[o]->getTensorDesc().getPrecision() == Precision::BF16) {
                    // if some layer (e.g. memory) consumes tensor, but must be fitted with another layer (e.g. memory output)
                    // in the net, whe must prevent this tensor to be fp32 - marked
                    bool notToMarkFP32 = false;
                    for (auto consumer : getInputTo(inputTo.second->outData[o])) {
                        if (_skipmarking.find(consumer.second->type) !=
                            _skipmarking.end()) {
                            notToMarkFP32 = true;
                        }
                    }
                    if (notToMarkFP32) {
                        continue;
                    }
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
        if (getInputTo(data).size() == 1) {
            if (_initbf16.find(getInputTo(data).begin()->second->type) == _initbf16.end()) {
                marked = true;
            }
        } else {
            // get all consumers
            for (auto o : getInputTo(data)) {
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
void BF16Transformer::addLayerToCNNNetworkAfterData(
        DataPtr parentOutData,
        CNNLayer::Ptr layer,
        const std::string& nextLayerName,
        ICNNNetwork& net,
        const int childInsDataIndex) {
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
        THROW_IE_EXCEPTION << "unexpected network type";
    }

    CNNLayerPtr nextLayer;
    if (!nextLayerName.empty()) {
        netImpl->getLayerByName(nextLayerName.c_str(), nextLayer, nullptr);
    }

    if (layer && (nextLayerName.empty() || (parentOutData == nullptr) || (childInsDataIndex != -1) ||
                  (getInputTo(parentOutData).find(nextLayerName) != getInputTo(parentOutData).end()))) {
        auto getTensorDesc = [](CNNLayerPtr& nextLayer) {
            const DataPtr insData = nextLayer->insData[0].lock();
            return insData->getTensorDesc();
        };

        const TensorDesc& parentTensorDesc = parentOutData != nullptr ? parentOutData->getTensorDesc() : getTensorDesc(nextLayer);
        DataPtr newEdgeAfterLayer(new Data(layer->name, parentTensorDesc));
        newEdgeAfterLayer->setName(layer->name);
        getCreatorLayer(newEdgeAfterLayer) = layer;
        getInputTo(newEdgeAfterLayer).clear();


        if (netImpl == nullptr) {
            THROW_IE_EXCEPTION << "unexpected network type";
        }
        netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
        IE_SUPPRESS_DEPRECATED_START
        netImpl->addLayer(layer);
        IE_SUPPRESS_DEPRECATED_END

        if (parentOutData != nullptr) {
            getInputTo(parentOutData)[layer->name] = layer;
            layer->insData.push_back(parentOutData);
        }
        layer->outData.push_back(newEdgeAfterLayer);

        if (!nextLayerName.empty()) {
            // CNNLayerPtr nextLayer = getInputTo(parentOutData)[nextLayerName];
            getInputTo(newEdgeAfterLayer)[nextLayerName] = nextLayer;

            if (parentOutData != nullptr) {
                getInputTo(parentOutData).erase(nextLayerName);

                if (childInsDataIndex == -1) {
                    for (size_t i = 0; i < nextLayer->insData.size(); i++) {
                        if (nextLayer->insData[i].lock() == parentOutData) {
                            nextLayer->insData[i] = newEdgeAfterLayer;
                        }
                    }
                } else {
                    nextLayer->insData[childInsDataIndex] = newEdgeAfterLayer;
                }
            } else {
                nextLayer->insData.push_back(newEdgeAfterLayer);
            }
        } else {
            CNNLayerPtr parent = getCreatorLayer(parentOutData).lock();
            if (parent == nullptr) {
                THROW_IE_EXCEPTION << "parent data is absent";
            }
            netImpl->removeOutput(parent->name);
            netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
            netImpl->addOutput(layer->name);
        }
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void BF16Transformer::insertConvertAfterInput(InferenceEngine::CNNNetwork &network) {
    auto inputLayers = InferenceEngine::CNNNetGetAllInputLayers(network);
    for (auto inputIter : inputLayers) {
        for (size_t o = 0; o < inputIter->outData.size(); o++) {
            for (auto bfInitIter : getInputTo(inputIter->outData[o])) {
                if (inputIter->outData[o]->getPrecision() == Precision::BF16) {
                    // we don't need to enforce bf16-mode for the next layer
                    break;
                }
                auto bfInitLayer = bfInitIter.second;
                if (_initbf16.find(bfInitLayer->type) != _initbf16.end()) {
                    if (CaselessEq<std::string>()(bfInitLayer->type, "convolution")) {
                        // TODO: have to be removed after adding suitable implementation for convolution
                        break;
                    }
                    // insert convert
                    std::string layerName = inputIter->outData[o]->getName();
                    LayerParams cnnLayerParams{layerName, "Convert", Precision::FP32};
                    auto lay = std::make_shared<InferenceEngine::CNNLayer>(cnnLayerParams);
                    std::map<std::string, std::string> par = {{"name",      layerName},
                                                              {"type",      "Convert"},
                                                              {"precision", "FP32"}};
                    lay->params = par;
                    CNNLayerPtr convertLayer(lay);
                    BF16Transformer::addLayerToCNNNetworkAfterData(inputIter->outData[o], convertLayer, bfInitLayer->name,
                                                                   network);
                    // compute input port id for bfInitLayer
                    for (size_t i = 0; i < bfInitLayer->insData.size(); i++) {
                        if (bfInitLayer->insData[i].lock()->getName() == inputIter->outData[o]->getName()) {
                            // set conv input as bf
                            bfInitLayer->insData[i].lock()->setPrecision(Precision::BF16);
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }
}