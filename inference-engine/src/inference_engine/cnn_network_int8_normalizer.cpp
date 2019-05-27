// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_int8_normalizer.hpp"

#include <vector>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <ie_common.h>
#include <details/ie_cnn_network_tools.h>
#include <details/caseless.hpp>
#include <blob_factory.hpp>
#include <data_stats.h>
#include "cnn_network_impl.hpp"
#include "cnn_network_stats_impl.hpp"
#include "debug.h"
#include <fstream>
#include "ie_util_internal.hpp"
#include <utility>


using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

using StatsMap = std::map<std::string, InferenceEngine::NetworkNodeStatsPtr>;


CNNStatisticHelper::CNNStatisticHelper(CNNNetwork &network, const std::map<std::string, NetworkNodeStatsPtr> &internalNodesStats,
                                       int maxSign, int maxUnsign) {
    internalNodesStats_ = internalNodesStats;
    network_ = network;
    maxSign_ = maxSign;
    maxUnsign_ = maxUnsign;

    NormalizeStatistic();
}

bool CNNStatisticHelper::canLayerBeQuantized(CNNLayer::Ptr layer) const {
    // verification of existing statistic for all inputs
    for (const auto i : layer->insData) {
        if (internalNodesStats_.find(i.lock()->creatorLayer.lock()->name) == internalNodesStats_.end()) {
            return false;
        }
    }
    // verification if there is a statistic for output of the layer
    if ((layer->outData.size() > 1) && (internalNodesStats_.find(layer->name) == internalNodesStats_.end())) {
        return false;
    }
    return true;
}

void CNNStatisticHelper::copyStatistics(const std::string& srcName, const std::string& dstName) {
    internalNodesStats_[dstName] = internalNodesStats_[srcName];
}

bool CNNStatisticHelper::hasNegativeOutput(const std::string &layerName, int outputPort) const {
    // TODO(amalyshe) parameter outputPort is not used yet, logic of dedication to the port
    // should be implemented

    NetworkNodeStatsPtr layerStat = internalNodesStats_.at(layerName);
    for (auto v : layerStat->_minOutputs) {
        if (v < 0.f) {
            return true;
        }
    }
    return false;
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::getInputScale(CNNLayer::Ptr layer) const {
    auto previousLayer = layer->insData[0].lock()->creatorLayer.lock();
    std::string inputLayerName = previousLayer->name;

    // for case when we have the only average pooling before, we need to take this
    // statistic from input of avg pooling to compensate work of average pooling
    // and to stay in int8 as much as we can
    if (previousLayer->type == "Pooling" && (previousLayer->precision == Precision::I8 || previousLayer->precision == Precision::U8)) {
        // take input name to the pooling
        inputLayerName = previousLayer->insData[0].lock()->creatorLayer.lock()->name;
    }
    size_t inputChannels = layer->insData[0].lock()->getTensorDesc().getDims()[1];
    if (getStatistic(previousLayer)->_minOutputs.size() != inputChannels
        || getStatistic(previousLayer)->_maxOutputs.size() != inputChannels) {
        THROW_IE_EXCEPTION << "min and max sizes should be equal to input channels count for " << previousLayer->name;
    }

    return calculateScaleFactor(inputChannels, getStatistic(previousLayer),
                                hasNegativeOutput(previousLayer->name) ? maxSign_ : maxUnsign_);
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::getOutputScale(CNNLayer::Ptr layer) const {
    // TODO(amalyshe) for now we are looking to precision on the data node
    size_t outputChannels = layer->outData[0]->getTensorDesc().getDims()[1];
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Trying to get scales after layer having multiple output ports";
    }
    if (getStatistic(layer)->_minOutputs.size() != outputChannels
        || getStatistic(layer)->_maxOutputs.size() != outputChannels) {
        THROW_IE_EXCEPTION << "min and max sizes should be equal to output channels count for " << layer->name;
    }

    return calculateScaleFactor(outputChannels, getStatistic(layer),
                                layer->outData[0]->getPrecision() == Precision::I8 ? maxSign_ : maxUnsign_);
}

int CNNStatisticHelper::getMaxSignValue() const {
    return maxSign_;
}

InferenceEngine::Blob::Ptr CNNStatisticHelper::calculateScaleFactor(size_t channels ,
    NetworkNodeStatsPtr stats, int maxInt) const {
    if (stats->_minOutputs.size() != channels || stats->_maxOutputs.size() != channels) {
        THROW_IE_EXCEPTION << "min and max sizes should be equal to channels count";
    }

    // Creating i-scale blob
    std::shared_ptr<Data> iScaleData = std::shared_ptr<Data>(new Data("scale", { channels }, Precision::FP32, Layout::C));
    auto iScale = CreateBlobFromData(iScaleData);
    iScale->allocate();
    float* iScaleMemory = static_cast<float*>(iScale->buffer());

    for (int c = 0; c < channels; c++) {
        float maxc = 0;
            // maxc = fmax(maxc, fabs(stats[k]->_minOutputs[c]));        // TODO Check if we should take minimums into account
            maxc = fmax(maxc, fabs(stats->_maxOutputs[c]));
            maxc = fmax(maxc, fabs(stats->_minOutputs[c]));

        iScaleMemory[c] = maxc / static_cast<float>(maxInt);

        if (fabs(iScaleMemory[c]) < 1e-7) {
            iScaleMemory[c] = 1.0f;
        }
    }
    return iScale;
}

NetworkNodeStatsPtr CNNStatisticHelper::getStatistic(CNNLayer::Ptr layer) const {
    // TODO(amalyshe) all logic of traversing over network and get apropriate statistics should be here
    // for now it is a stub
    auto it = internalNodesStats_.find(getLatestInFuse(layer)->name);
    if (it != internalNodesStats_.end()) {
        return it->second;
    }
    THROW_IE_EXCEPTION << "no stat for layer " << getLatestInFuse(layer)->name;
}

CNNLayer::Ptr CNNStatisticHelper::getLatestInFuse(CNNLayer::Ptr layer) const {
    if (layer->outData[0]->inputTo.size() == 1 &&
        (CaselessEq<std::string>()(layer->outData[0]->inputTo.begin()->second->type, "relu") ||
         CNNNetworkInt8Normalizer::isReLULikeClamp(layer->outData[0]->inputTo.begin()->second)))  {
        return layer->outData[0]->inputTo.begin()->second;
    }
    // Conv-Sum-ReLU fuse
    // We need to return original layer if it will be used as a sum parame and ReLU if
    // iterating over outputs of pointed layer and look for the only eltwise
    CNNLayer::Ptr eltwise = nullptr;
    if (layer->outData.size() == 1) {
        for (auto it : layer->outData[0]->inputTo) {
            if (CaselessEq<std::string>()(it.second->type, "eltwise")) {
                if (eltwise) {
                    THROW_IE_EXCEPTION << "Pattern when one layer pass data to several eltwise layers are not supported in int8 quantization";
                }
                eltwise = it.second;
            }
        }
    }

    if (eltwise) {
        // if current layer is not a convolution return it as finish of fuse
        if (!CaselessEq<std::string>()(layer->type, "convolution")) {
            return layer;
        } else {
            // look to the ports of eltwise
            if (eltwise->insData[1].lock()->creatorLayer.lock() == layer &&
                CaselessEq<std::string>()(eltwise->insData[0].lock()->creatorLayer.lock()->type, "convolution") &&
                eltwise->insData[0].lock()->inputTo.size() == 1) {
                // this is a case when two convolutions come to eltwise, the second one will be selected for fuse,
                // first will be used as sum operator
                return layer;
            }
            // given layer is a convolution and will be used for fuse, but we need to verify if there is ReLU after eltwise
            if (eltwise->outData[0]->inputTo.size() == 1 &&
                (CaselessEq<std::string>()(eltwise->outData[0]->inputTo.begin()->second->type, "relu") ||
                 CNNNetworkInt8Normalizer::isReLULikeClamp(eltwise->outData[0]->inputTo.begin()->second))) {
                return eltwise->outData[0]->inputTo.begin()->second;
            }
            return eltwise;
        }
    }

    return layer;
}


void CNNStatisticHelper::NormalizeStatistic() {
    StatsMap newMap;

    float dummy = 0.0f;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network_);
    for (auto l : sortedLayers) {
        // if layer's statistic exists in the newMap, ignore it
        if (newMap.find(l->name) != newMap.end()) {
            continue;
        }
        // verify if layer is starter layer for propagating of statistic
        bool isStarterLayer = false;

        // a case if we do not have converted statistic before the current layer
        // go over all inputs and verify if statistic exists for all of inputs
        bool allInputsHaveStatistics = true;
        for (auto i : l->insData) {
            if (newMap.find(i.lock()->creatorLayer.lock()->name) == newMap.end()) {
                allInputsHaveStatistics = false;
                break;
            }
        }
        // if we do not have statistic - verify who is consumer of this layer
        if (!allInputsHaveStatistics) {
            if (l->outData.size() == 1) {
                for (auto it : l->outData[0]->inputTo) {
                    if (CaselessEq<std::string>()(it.second->type, "scaleshift") ||
                        CaselessEq<std::string>()(it.second->type, "convolution")) {
                        isStarterLayer = true;
                        break;
                    }
                }
            }
        } else {
            isStarterLayer = true;
        }
        if (CaselessEq<std::string>()(l->type, "scaleshift") ||
            CaselessEq<std::string>()(l->type, "convolution")) {
            isStarterLayer = true;
        }

        if (!isStarterLayer) {
            continue;
        }

        // we do not support yet layers for quantization which split data
        if (l->outData.size() != 1) {
            continue;
        }

        InferenceEngine::NetworkNodeStatsPtr currentStat = std::make_shared<NetworkNodeStats>();

        bool perChannelScale = true;


        if (CaselessEq<std::string>()(l->type, "concat")
            && l->outData.size() == 1
            && l->outData[0]->getTensorDesc().getDims().size() == 4
            && allInputsHaveStatistics) {
            size_t concatLayerIdx = 0;
            for (int k = 0; k < l->insData.size(); k++) {
                auto prevKLayer = l->insData[k].lock()->creatorLayer.lock();
                // looking for the statistic for prevKLayer
                auto kLayerStat = newMap.find(prevKLayer->name);
                if (kLayerStat != newMap.end()) {
                    for (size_t ikStat = 0; ikStat < kLayerStat->second->_maxOutputs.size(); ikStat++, concatLayerIdx++) {
                        currentStat->_maxOutputs.push_back(kLayerStat->second->_maxOutputs[ikStat]);
                        currentStat->_minOutputs.push_back(kLayerStat->second->_minOutputs[ikStat]);
                    }
                } else {
                    THROW_IE_EXCEPTION << "We have incomplete statistic for predecessors of concat layer " << l->name;
                }
            }
        } else if (CaselessEq<std::string>()(l->type, "resample")) {
            if (l->insData.size() == 1) {
                CNNLayerPtr creator = l->insData[0].lock()->getCreatorLayer().lock();
                if (CaselessEq<std::string>()(creator->type, "concat")) {
                    auto concatStat = newMap[creator->name];
                    currentStat->_maxOutputs = concatStat->_maxOutputs;
                    currentStat->_minOutputs = concatStat->_minOutputs;
                    newMap[l->name] = currentStat;
                } else {
                    auto itOld = internalNodesStats_.find(l->name);
                    if (itOld != internalNodesStats_.end()) {
                        currentStat->_maxOutputs = itOld->second->_maxOutputs;
                        currentStat->_minOutputs = itOld->second->_minOutputs;
                        newMap[l->name] = currentStat;
                    }
                }
            }
        } else {
            // go over all children until we get convoluition, scaleshift, eltwise or unknown layer
            // layers Pooling and ReLU are passthrough
            // to understand the granularity of the scaling
            // layer concat is a layer which produce statistics and waterfall it down
            std::vector<CNNLayer::Ptr> toAnalyze;
            for (auto it : l->outData[0]->inputTo) {
                toAnalyze.push_back(it.second);
            }

            if (CaselessEq<std::string>()(l->type, "eltwise")) {
                perChannelScale = false;
            }
            while (!toAnalyze.empty() && perChannelScale) {
                CNNLayer::Ptr tl = toAnalyze.back();
                toAnalyze.pop_back();
                if (CaselessEq<std::string>()(tl->type, "pooling") ||
                    CaselessEq<std::string>()(tl->type, "relu") ||
                    CNNNetworkInt8Normalizer::isReLULikeClamp(tl) ||
                    CaselessEq<std::string>()(tl->type, "concat")) {
                    if (tl->outData.size() == 1) {
                        for (auto it : tl->outData[0]->inputTo) {
                            toAnalyze.push_back(it.second);
                        }
                    }
                } else if (CaselessEq<std::string>()(tl->type, "convolution")) {
                    // verify number of groups
                    ConvolutionLayer *pConv = dynamic_cast<ConvolutionLayer *>(tl.get());
                    if (pConv == nullptr) {
                        THROW_IE_EXCEPTION << "Layer " << tl->name << " is not instance of ConvolutionLayer class";
                    }
                    if (pConv->_group != pConv->_out_depth) {
                        perChannelScale = false;
                    }
                } else if (CaselessEq<std::string>()(tl->type, "eltwise")) {
                    perChannelScale = false;
                }
            }

            auto itOld = internalNodesStats_.find(getLatestInFuse(l)->name);
            if (itOld == internalNodesStats_.end()) {
                itOld = internalNodesStats_.find(l->name);
            }
            if (itOld != internalNodesStats_.end()) {
                if (!perChannelScale) {
                    currentStat->_maxOutputs.resize(itOld->second->_maxOutputs.size());
                    if (!itOld->second->_maxOutputs.empty()) {
                        float max = FLT_MIN;
                        DataStats::GetDataAbsMax(&itOld->second->_maxOutputs[0], itOld->second->_maxOutputs.size(), max);
                        std::fill(currentStat->_maxOutputs.begin(), currentStat->_maxOutputs.end(), max);
                    }

                    currentStat->_minOutputs.resize(itOld->second->_minOutputs.size());
                    if (!itOld->second->_minOutputs.empty()) {
                        float min = FLT_MAX;
                        DataStats::GetDataMinMax(&itOld->second->_minOutputs[0], itOld->second->_minOutputs.size(), min, dummy);
                        std::fill(currentStat->_minOutputs.begin(), currentStat->_minOutputs.end(), min);
                    }
                } else {
                    currentStat->_maxOutputs = itOld->second->_maxOutputs;
                    currentStat->_minOutputs = itOld->second->_minOutputs;
                }
            }


            if (l->outData.size() == 1) {
                size_t outputChannels = l->outData[0]->getTensorDesc().getDims()[1];
                auto oldStat = internalNodesStats_.find(l->name);
                if ((oldStat != internalNodesStats_.end()) && outputChannels > 1 && oldStat->second->_minOutputs.size() == 1) {
                    auto min = oldStat->second->_minOutputs[0];
                    auto max = oldStat->second->_maxOutputs[0];

                    currentStat->_minOutputs = std::vector<float>(outputChannels);
                    currentStat->_maxOutputs = std::vector<float>(outputChannels);
                    std::fill(currentStat->_minOutputs.begin(), currentStat->_minOutputs.end(), min);
                    std::fill(currentStat->_maxOutputs.begin(), currentStat->_maxOutputs.end(), max);
                }
            }
        }

        // propagate this statistic to all layers without scale in primitives
        if (!currentStat->_maxOutputs.empty() && !currentStat->_minOutputs.empty()) {
            std::vector<CNNLayer::Ptr> toAnalyze;
            toAnalyze.push_back(l);
            while (!toAnalyze.empty()) {
                CNNLayer::Ptr tl = toAnalyze.back();
                toAnalyze.pop_back();
                newMap[tl->name] = currentStat;
                if (tl->outData.size() == 1) {
                    for (auto it : tl->outData[0]->inputTo) {
                        if (CaselessEq<std::string>()(it.second->type, "pooling") ||
                                CaselessEq<std::string>()(it.second->type, "relu") ||
                                CNNNetworkInt8Normalizer::isReLULikeClamp(it.second)) {
                            toAnalyze.push_back(it.second);
                        }
                    }
                }
            }
        }
    }

    internalNodesStats_ = newMap;
}

void CNNNetworkInt8Normalizer::AddLayerToCNNNetworkBeforeLayer(CNNLayer::Ptr newLayer, CNNLayer::Ptr successor, size_t port) {
    // verify if data exists
    if (newLayer && successor && successor->insData.size() > port) {
        // get the insData
        DataPtr pData = successor->insData[port].lock();

        Data *edge2 = new Data(*pData.get());
        DataPtr newEdge(edge2);
        newEdge->getInputTo().clear();
        newEdge->getInputTo()[successor->name] = successor;
        newEdge->name = newLayer->name;
        newEdge->getCreatorLayer() = newLayer;
        successor->insData[port] = newEdge;
        newLayer->outData.push_back(newEdge);

        newLayer->insData.push_back(pData);
        pData->getInputTo().erase(successor->name);
        pData->getInputTo()[newLayer->name] = newLayer;
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void CNNNetworkInt8Normalizer::AddLayerToCNNNetworkAfterData(DataPtr pData, CNNLayer::Ptr layer, const std::string& nextLayerName) {
    // verify if data exists
    if (pData && layer && pData->creatorLayer.lock() && pData->inputTo.find(nextLayerName) != pData->inputTo.end()) {
        CNNLayerPtr nextLayer = pData->inputTo[nextLayerName];

        DataPtr newEdgeAfterLayer(new Data(*pData.get()));
        newEdgeAfterLayer->name = layer->name;
        newEdgeAfterLayer->creatorLayer = layer;
        newEdgeAfterLayer->inputTo.clear();
        newEdgeAfterLayer->inputTo[nextLayerName] = nextLayer;
        newEdgeAfterLayer->setPrecision(Precision::FP32);

        pData->getInputTo().erase(nextLayerName);
        pData->getInputTo()[layer->name] = layer;

        layer->insData.push_back(pData);
        layer->outData.push_back(newEdgeAfterLayer);

        for (size_t i = 0; i < nextLayer->insData.size(); i++) {
            if (nextLayer->insData[i].lock() == pData) {
                nextLayer->insData[i] = newEdgeAfterLayer;
            }
        }
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void CNNNetworkInt8Normalizer::fillInScaleShift(ScaleShiftLayer* scshLayer, size_t c, float* weightsN, float* weightsD) {
    // Setting "scales"
    SizeVector weightsSize = { c };
    TensorDesc weightsDesc(Precision::FP32, weightsSize, InferenceEngine::C);
    scshLayer->_weights = InferenceEngine::make_shared_blob<float>(weightsDesc);
    scshLayer->_weights->allocate();
    float * weightsData = scshLayer->_weights->buffer();
    for (size_t i = 0; i < c; i++) {
        if (weightsN == nullptr && weightsD != nullptr) {
            weightsData[i] = 1.0 / weightsD[i];
        } else if (weightsD == nullptr && weightsN != nullptr) {
            weightsData[i] = weightsN[i];
        } else if (weightsN != nullptr && weightsD != nullptr) {
            weightsData[i] = weightsN[i] / weightsD[i];
        } else {
            weightsData[i] = 1.0;
        }
    }

    // Setting "shifts"
    SizeVector shiftsSize = { c };
    TensorDesc shiftsDesc(Precision::FP32, shiftsSize, InferenceEngine::C);
    scshLayer->_biases = InferenceEngine::make_shared_blob<float>(shiftsDesc);
    scshLayer->_biases->allocate();
    float * biasesData = scshLayer->_biases->buffer();
    for (size_t i = 0; i < c; i++) {
        biasesData[i] = 0.f;  // Setting to constant "0"
    }
}

void CNNNetworkInt8Normalizer::AddScaleShiftBetween(CNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
    CNNStatisticHelper& statHelper) {

    if (CaselessEq<std::string>()(layer2->type, "priorbox") ||
        CaselessEq<std::string>()(layer2->type, "priorboxclustered")) {
        return;
    }

    // Searching the connection between the layers
    int l1_out_i = 0;
    for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
        if (layer1->outData[l1_out_i]->inputTo.find(layer2->name) != layer1->outData[l1_out_i]->inputTo.end()) {
            break;
        }
    }
    if (l1_out_i == layer1->outData.size()) {
        THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
    }

    int l2_in_i = 0;
    for (; l2_in_i < layer2->insData.size(); l2_in_i++) {
        if (layer2->insData[l2_in_i].lock()->creatorLayer.lock() == layer1) {
            break;
        }
    }
    if (l2_in_i == layer2->insData.size()) {
        THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " inputs";
    }

    DataPtr outData = layer1->outData[l1_out_i];

    Blob::Ptr oScaleBlob = nullptr;
    if (layer1->blobs.find("o-scale") != layer1->blobs.end()) {
        oScaleBlob = layer1->blobs["o-scale"];
    }

    Blob::Ptr iScaleBlob = nullptr;
    if (layer2->blobs.find("i-scale") != layer2->blobs.end()) {
        iScaleBlob = layer2->blobs["i-scale"];
    }

    if (iScaleBlob == nullptr && oScaleBlob == nullptr) {
        return;  // No multipliers found around this edge. We can't create a ScaleShift here;
    } else {
        // Creating a ScaleShiftLayer
        std::string prefix;
        float *iScaleBuffer = nullptr, *oScaleBuffer = nullptr;
        if (oScaleBlob != nullptr) {
            oScaleBuffer = static_cast<float*>(oScaleBlob->buffer());
            prefix += "o";
        }
        if (iScaleBlob != nullptr) {
            iScaleBuffer = static_cast<float*>(iScaleBlob->buffer());
            prefix += "i";
        }

        std::string layerName = layer1->name + "_" + prefix + "ScaleShift_" + layer2->name;
        LayerParams ssCnnLayerParams{ layerName, "ScaleShift", Precision::FP32 };
        CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

        AddLayerToCNNNetworkAfterData(outData, ssCnnLayer, layer2->name);

        size_t c = static_cast<size_t>(outData->getDims()[1]);

        {
            ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());
            if (scshLayer == nullptr) {
                THROW_IE_EXCEPTION << "Layer " << ssCnnLayer->name << " is not instance of ScaleShiftLayer class";
            }
            fillInScaleShift(scshLayer, c, oScaleBuffer, iScaleBuffer);
        }

        Precision odPrecision = Precision::FP32;
        if (layer2->precision == Precision::I8) {
            odPrecision = statHelper.hasNegativeOutput(layer1->name) ? Precision::I8 : Precision::U8;
        }
        ssCnnLayer->outData[0]->setPrecision(odPrecision);
    }
}

void CNNNetworkInt8Normalizer::AddScaleShifts(CNNNetwork& net, CNNStatisticHelper& statHelper) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>> pairs;

    for (auto iter : sortedLayers) {
        for (int l1_out_i = 0; l1_out_i < iter->outData.size(); l1_out_i++) {
            for (auto nextIter : iter->outData[l1_out_i]->inputTo) {
                CNNLayer::Ptr next = nextIter.second;

                // Checking for an INT8 convolution or fully connected with FP32 output
                if ((CaselessEq<std::string>()(iter->type, "Convolution") ||
                     CaselessEq<std::string>()(iter->type, "FullyConnected")) &&
                    iter->precision == Precision::I8 &&
                    next->precision == Precision::FP32 &&
                    iter->outData[l1_out_i]->getPrecision() == Precision::FP32) {
                    // Do nothing here only if iter provides data to fp32 layers
                    // MKLDNNPlugin will generate x8->f32 convolution

                } else if ((iter->precision != Precision::FP32 && next->precision == Precision::FP32) ||
                           (iter->precision == Precision::FP32 && next->precision != Precision::FP32)) {
                    pairs.push_back(std::pair<CNNLayerPtr, CNNLayerPtr>(iter, next));
                }
            }
        }
    }

    for (auto& pair : pairs) {
        AddScaleShiftBetween(net, pair.first, pair.second, statHelper);
    }
}

void CNNNetworkInt8Normalizer::ClampsToReLU(CNNNetwork& net, CNNStatisticHelper& statHelper) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    for (auto iter : sortedLayers) {
        if (isReLULikeClamp(iter) && (iter->precision == Precision::I8 || iter->precision == Precision::U8)) {
            std::string layerName = iter->name + "_ReLU";
            LayerParams ssCnnLayerParams{ layerName, "ReLU", iter->precision };
            CNNLayerPtr ssCnnLayer(new ReLULayer(ssCnnLayerParams));

            auto previousLayer = iter->insData[0].lock()->creatorLayer.lock();
            ssCnnLayer->insData.push_back(iter->insData[0]);
            ssCnnLayer->insData[0].lock()->inputTo.erase(iter->name);
            ssCnnLayer->insData[0].lock()->inputTo[iter->name] = ssCnnLayer;

            ssCnnLayer->outData.push_back(iter->outData[0]);
            ssCnnLayer->outData[0]->creatorLayer = ssCnnLayer;

            iter->insData.clear();
            iter->outData.clear();
        }
    }
}

void CNNNetworkInt8Normalizer::ScaleDataToInt(const float* srcData, size_t srcSize, Blob::Ptr int8blob, const std::vector<float>& scales) {
    if (scales.size() == 0 || /*srcblob->size()*/srcSize % scales.size() != 0) {
        THROW_IE_EXCEPTION << "Wrong number of scale factors";
    }

    size_t channels = scales.size();
    size_t channelSize = /*srcblob->size()*/srcSize / channels;

    const float* data = srcData;
    if (int8blob->getTensorDesc().getPrecision() == Precision::I8) {
        int8_t* int8data = static_cast<int8_t*>(int8blob->buffer());
        int minValue = std::numeric_limits<int8_t>::min();
        int maxValue = std::numeric_limits<int8_t>::max();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
                    val = maxValue;
                } else if (val < minValue) {
                    val = minValue;
                }

                int8data[offset + i] = round(val);
            }
        }
    } else if (int8blob->getTensorDesc().getPrecision() == Precision::I32) {
        int32_t* int32data = static_cast<int32_t*>(int8blob->buffer());
        int maxValue = std::numeric_limits<int32_t>::max();
        int minValue = std::numeric_limits<int32_t>::min();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
                    val = maxValue;
                } else if (val < minValue) {
                    val = minValue;
                }

                int32data[offset + i] = round(val);
            }
        }
    }
}

CNNLayer::Ptr CNNNetworkInt8Normalizer::createDWConvolutionForScale(const std::string &layerName, size_t channels, float *ssWValues, float *ssSValues) {
    // create new Convolution layer
    LayerParams params;
    params.name = layerName;
    params.precision = Precision::FP32;
    params.type = "Convolution";

    CNNLayerPtr lptr = std::make_shared<ConvolutionLayer>(params);
    auto *pConv = dynamic_cast<ConvolutionLayer *>(lptr.get());
    if (pConv == nullptr) {
        THROW_IE_EXCEPTION << "Layer " << lptr->name << " is not instance of ConvolutionLayer class";
    }

    pConv->_kernel.insert(X_AXIS, 1);
    pConv->_kernel.insert(Y_AXIS, 1);
    pConv->_stride.insert(X_AXIS, 1);
    pConv->_stride.insert(Y_AXIS, 1);
    pConv->_padding.insert(X_AXIS, 0);
    pConv->_padding.insert(Y_AXIS, 0);
    pConv->_pads_end.insert(X_AXIS, 0);
    pConv->_pads_end.insert(Y_AXIS, 0);
    pConv->_dilation.insert(X_AXIS, 1);
    pConv->_dilation.insert(Y_AXIS, 1);

    pConv->_out_depth = channels;
    // mkl-dnn does not have i8 depthwise convolution accepting signed i8 input
    // when it is available, need to uncomment below lines

    // workaround - creation of new weights for simple convolution
    if (pConv->_out_depth % 16 == 0) {
        pConv->_group = pConv->_out_depth / 16;
        Blob::Ptr weights = nullptr;
        std::shared_ptr<Data> wData = std::shared_ptr<Data>(new Data("weights", { pConv->_out_depth * 16 }, Precision::FP32, Layout::C));
        weights = CreateBlobFromData(wData);
        weights->allocate();
        float *buffer = weights->buffer().as<float *>();
        size_t iDist = 0, iSrc = 0;
        for (size_t g = 0; g < pConv->_group; g++) {
            for (size_t k = 0; k < 16; k++) {
                for (size_t s = 0; s < 16; s++) {
                    buffer[iDist++] = (s == k) ? ssWValues[iSrc++] : 0.f;
                }
            }
        }
        pConv->_weights = weights;
        pConv->blobs["weights"] = weights;
    } else {
        Blob::Ptr weights = nullptr;
        std::shared_ptr<Data> wData = std::shared_ptr<Data>(new Data("weights", { pConv->_out_depth * pConv->_out_depth }, Precision::FP32, Layout::C));
        weights = CreateBlobFromData(wData);
        weights->allocate();
        float *buffer = weights->buffer().as<float *>();
        for (size_t i = 0, idx = 0; i < pConv->_out_depth; i++) {
            for (size_t j = 0; j < pConv->_out_depth; j++) {
                if (i == j) {
                    buffer[idx] = ssWValues[i];
                } else {
                    buffer[idx] = 0.f;
                }
                idx++;
            }
        }
        pConv->_weights = weights;
        pConv->blobs["weights"] = weights;
        pConv->_group = 1;
    }
    // end of workaround

    // fililng of biases
    Blob::Ptr biasesBlob = nullptr;
    std::shared_ptr<Data> bData = std::shared_ptr<Data>(new Data("biases", { pConv->_out_depth }, Precision::FP32, Layout::C));
    biasesBlob = CreateBlobFromData(bData);
    biasesBlob->allocate();
    float *bufferBiases = biasesBlob->buffer().as<float *>();
    for (size_t c = 0; c < pConv->_out_depth; c++) {
        bufferBiases[c] = ssSValues[c];
    }
    pConv->_biases = biasesBlob;

    pConv->blobs["weights"] = pConv->_weights;
    pConv->blobs["biases"] = pConv->_biases;
    return lptr;
}

void CNNNetworkInt8Normalizer::replaceScaleShiftByDWConvolution(CNNNetwork &net) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);
    for (auto layer : sortedLayers) {
        if (CaselessEq<std::string>()(layer->type, "scaleshift")
            && layer->insData[0].lock()->creatorLayer.lock()
            && !CaselessEq<std::string>()(layer->insData[0].lock()->creatorLayer.lock()->type, "input")
            && layer->outData[0]->inputTo.size() > 0) {
            const auto dims = layer->insData[0].lock()->getTensorDesc().getDims();
            // only four or five dimensions Convolution layers are supported
            if ((dims.size() == 4) || (dims.size() == 5)) {
                // verification if this layer does not pass data to PriorBox, if it passes, we do not substitute
                bool notToPriorBox = true;
                for (auto o : layer->outData[0]->inputTo) {
                    if (CaselessEq<std::string>()(o.second->type, "priorbox") ||
                        CaselessEq<std::string>()(o.second->type, "priorboxclustered")) {
                        notToPriorBox = false;
                    }
                }
                if (notToPriorBox) {
                    ScaleShiftLayer *pSS = dynamic_cast<ScaleShiftLayer *>(layer.get());
                    float *ssWValues = pSS->_weights->buffer().as<float *>();
                    float *ssSValues = pSS->_biases->buffer().as<float *>();
                    CNNLayer::Ptr newLayer = createDWConvolutionForScale(layer->name, layer->outData[0]->getTensorDesc().getDims()[1], ssWValues, ssSValues);

                    newLayer->outData = layer->outData;
                    newLayer->outData[0]->creatorLayer = newLayer;
                    newLayer->insData = layer->insData;
                    newLayer->insData[0].lock()->inputTo.erase(layer->name);
                    newLayer->insData[0].lock()->inputTo[newLayer->name] = newLayer;
                }
            }
        }
    }
}

void CNNNetworkInt8Normalizer::QuantizeConvolutionOrFullyConnected(CNNLayer::Ptr convolution,
                                                    CNNStatisticHelper& statHelper) {
    size_t inputChannels = convolution->insData[0].lock()->getTensorDesc().getDims()[1];
    size_t outputChannels = convolution->outData[0]->getTensorDesc().getDims()[1];

    auto iScale = statHelper.getInputScale(convolution);

    convolution->blobs["i-scale"] = iScale;

    Blob::Ptr weights = nullptr;
    Blob::Ptr biases = nullptr;

    Blob::Ptr int8weights = nullptr;
    Blob::Ptr int32biases = nullptr;

    if (convolution->blobs.find("weights")!= convolution->blobs.end()) {
        weights = convolution->blobs["weights"];

        // Creating int8 weights blob
        std::shared_ptr<Data> int8WeightsData = std::shared_ptr<Data>(new Data("weights", weights->dims(), Precision::I8, weights->layout()));
        int8weights = CreateBlobFromData(int8WeightsData);
        int8weights->allocate();
        convolution->blobs["weights"] = int8weights;
    }

    if (convolution->blobs.find("biases")!= convolution->blobs.end()) {
        biases = convolution->blobs["biases"];

        // Creating int8 biases blob
        std::shared_ptr<Data> int32BiasesData = std::shared_ptr<Data>(new Data("biases", biases->dims(), Precision::I32, biases->layout()));
        int32biases = CreateBlobFromData(int32BiasesData);
        int32biases->allocate();
        convolution->blobs["biases"] = int32biases;
    }

    std::vector<float> weightScalers;


    // Creating w-scale blob
    if (weights) {
        const float *weight = static_cast<const float *>(weights->buffer());

        WeightableLayer *pConv = dynamic_cast<WeightableLayer *>(convolution.get());
        ConvolutionLayer *pConv1 = dynamic_cast<ConvolutionLayer *>(convolution.get());

        if (pConv1 != nullptr && pConv1->_group == 0) {
            THROW_IE_EXCEPTION << "Convolution '" << convolution->name << "'has wrong groups number == 0";
        }
        int group = 1;
        if (pConv1 != nullptr && pConv1->_group != 1) {
            group = pConv1->_group;
        }


        std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale

        size_t W_CO = outputChannels / group,
        W_CI = inputChannels / group,
        W_HW = weights->size()/ W_CI / W_CO / group;

        {
            float *iScaleMemory = static_cast<float *>(iScale->buffer());
            for (size_t g = 0; g < group; g++) {
                for (size_t co = 0; co < W_CO; co++) {
                    for (size_t ci = 0; ci < W_CI; ci++) {
                        size_t kernelBase = g * W_CO * W_CI * W_HW + co * W_CI * W_HW + ci * W_HW;
                        for (size_t hw = 0; hw < W_HW; hw++) {
                            newWeights.push_back(weight[kernelBase + hw] * iScaleMemory[g * W_CI + ci]);
                        }
                    }
                }
            }
        }
        size_t outChannelSize = weights->dims()[0] / W_CO / group;

        // Calculating weights normalization scale factor (w-scale)
        float *weight_convolution;
        size_t co;
        for (co = 0, weight_convolution = &newWeights[0]; co < outputChannels; co++, weight_convolution += outChannelSize) {
            float max = FLT_MIN;
            DataStats::GetDataAbsMax(weight_convolution, outChannelSize, max);

            float scaler = static_cast<float>(statHelper.getMaxSignValue())/ max;
            weightScalers.push_back(scaler);
        }

        std::shared_ptr<Data> wScaleData = std::shared_ptr<Data>(new Data("w-scale", { outputChannels }, Precision::FP32, Layout::C));
        auto wScale = CreateBlobFromData(wScaleData);
        wScale->allocate();

        float *wScaleMemory = static_cast<float *>(wScale->buffer());

        for (size_t i = 0; i < outputChannels; i++) {
            wScaleMemory[i] = 1.0 / weightScalers[i];
        }
        convolution->blobs["w-scale"] = wScale;

        auto oScale = statHelper.getOutputScale(statHelper.getLatestInFuse(convolution));
        convolution->blobs["o-scale"] = oScale;

        // debug scales. Need to compare with actual values in FP32 scoring
        convolution->blobs["ext-scale"] = convolution->blobs["o-scale"];

        // Normalizing the weights
        ScaleDataToInt(&newWeights[0], weights->size(), int8weights, weightScalers);
    }

    // Normalizing the biases
    if (biases) {
        const float *bias = static_cast<const float *>(biases->buffer());
        ScaleDataToInt(bias, biases->size(), int32biases, weightScalers);
    }
}

bool CNNNetworkInt8Normalizer::layerProducesFloat(const CNNLayer::Ptr layer) {
    // currently we support only case of layers which have one output port
    if (layer->outData.size() > 1) {
        return false;
    }

    bool consumersFP32 = true;
    for (const auto dOut : layer->outData[0]->inputTo) {
        if (dOut.second->precision != Precision::FP32) {
            consumersFP32 = false;
        }
    }
    return consumersFP32;
}

void CNNNetworkInt8Normalizer::returnTailToFP32(const CNNLayer::Ptr layer) {
    std::set<CNNLayer::Ptr> layersToReturn;
    if (layerProducesFloat(layer)) {
        layersToReturn.insert(layer);
    }

    while (!layersToReturn.empty()) {
        CNNLayer::Ptr layerA = *layersToReturn.begin();
        layersToReturn.erase(layerA);
        // 1. if it is Pooling layer, or concat layer, we can return it to FP32 as well
        // we need to return it's out data
        if ((CaselessEq<std::string>()(layerA->type, "pooling")
            || CaselessEq<std::string>()(layerA->type, "concat")) &&
            layerA->outData.size() == 1) {
            layerA->precision = Precision::FP32;
            layerA->outData[0]->setPrecision(Precision::FP32);
        }

        if ((CaselessEq<std::string>()(layerA->type, "convolution")
            || CaselessEq<std::string>()(layerA->type, "fullyconnected")
            || CaselessEq<std::string>()(layerA->type, "relu")
            || isReLULikeClamp(layerA)) &&
            layerA->outData.size() == 1) {
            layerA->outData[0]->setPrecision(Precision::FP32);
            if (CaselessEq<std::string>()(layerA->type, "relu")
                && isNextFusionAllowed(layerA->insData[0].lock()->creatorLayer.lock())) {
                layerA->precision = Precision::FP32;
                layerA->insData[0].lock()->creatorLayer.lock()->outData[0]->setPrecision(Precision::FP32);
            }
        }


        // adding parents for analysis
        if (!CaselessEq<std::string>()(layerA->type, "convolution") &&
            !CaselessEq<std::string>()(layerA->type, "fullyconnected")) {
            // for all parents, if they produce data to only FP32 layers
            for (auto i : layerA->insData) {
                DataPtr d = i.lock();
                if (d->creatorLayer.lock()->precision != Precision::FP32
                    && (CaselessEq<std::string>()(layerA->type, "pooling")
                        || CaselessEq<std::string>()(layerA->type, "relu")
                        || isReLULikeClamp(layerA)
                        || CaselessEq<std::string>()(layerA->type, "concat"))) {
                    if (layerProducesFloat(d->creatorLayer.lock())) {
                        layersToReturn.insert(d->creatorLayer.lock());
                    }
                }
            }
        }
    }
}

bool CNNNetworkInt8Normalizer::isNextFusionAllowed(const CNNLayer::Ptr& layer) {
    // fusion can happen only if initial layer supplies data to only one layer
    // if it sends to several layers - it is safe to execute initial layer in any precision
    if (layer->outData[0]->inputTo.size() == 1) {
        std::string aType = layer->outData[0]->inputTo.begin()->second->type;
        if (CaselessEq<std::string>()(aType, "relu")) {
            ReLULayer *rL = dynamic_cast<ReLULayer *>(layer->outData[0]->inputTo.begin()->second.get());
            if (rL == nullptr) {
                THROW_IE_EXCEPTION << "Layer " << layer->outData[0]->inputTo.begin()->second->name
                                   << " is not instance of ReLULayer class";
            }
            if (rL->negative_slope != 0.f) {
                return false;
            }
        } else if (CaselessEq<std::string>()(aType, "clamp")) {
            if (!isReLULikeClamp(layer->outData[0]->inputTo.begin()->second)) {
                return false;
            }
        } else {
            static const InferenceEngine::details::caseless_set<std::string> nonSuportedActivations =
            {"elu", "clamp", "tanh", "logistic", "square", "abs",
            "sqrt", "linear", "bounded_elu", "sort_relu", "relu6"};
            return !(nonSuportedActivations.find(aType) != nonSuportedActivations.end());
        }
    }
    return true;
}

bool CNNNetworkInt8Normalizer::isReLULikeClamp(CNNLayer::Ptr layer) {
    if (CaselessEq<std::string>()(layer->type, "Clamp")) {
        ClampLayer *clamp = dynamic_cast<ClampLayer *>(layer.get());
        if (clamp == nullptr) {
            THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer '" << layer->name << "' to Clamp";
        }
        return clamp->min_value == 0;
    }
    return false;
}

void CNNNetworkInt8Normalizer::DefinesExecutionPrecision(CNNNetwork &net, CNNStatisticHelper &statHelper) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    // Converting layers to Int8. Calculating the multipliers if needed
    for (auto iter : sortedLayers) {
        if (iter->params.find("quantization_level") != iter->params.end() && iter->params["quantization_level"] == "FP32") {
            continue;
        }

        // Legacy: FullyConnected should not be converted to Int8,
        // if it isn't explicitly marked to.
        if (iter->params.find("quantization_level") == iter->params.end() && CaselessEq<std::string>()(iter->type, "fullyconnected")) {
            continue;
        }

        if (!statHelper.canLayerBeQuantized(iter)) {
            continue;
        }

        if (CaselessEq<std::string>()(iter->type, "convolution") ||
            CaselessEq<std::string>()(iter->type, "fullyconnected")) {
            if (isNextFusionAllowed(iter)) {
                iter->precision = Precision::I8;
                // we will override I8 to U8 during analysing of Conv-ReLU and Conv-Sum-ReLU fusions
                iter->outData[0]->setPrecision(Precision::I8);
            }
        } else if (CaselessEq<std::string>()(iter->type, "relu") ||
                   isReLULikeClamp(iter)) {
            // casting to ReLU
            ReLULayer *rL = dynamic_cast<ReLULayer *>(iter.get());
            DataPtr outData = iter->outData.size() ? iter->outData[0] : nullptr;
            if (iter->insData[0].lock()->creatorLayer.lock()->precision != Precision::FP32
                && outData->getPrecision() == Precision::FP32) {
                iter->precision = Precision::I8;
                if (rL != nullptr && rL->negative_slope != 0.0f) {
                    outData->setPrecision(Precision::I8);
                } else {
                    outData->setPrecision(Precision::U8);
                    // if convolution is a predecessor, change its data to U8 also
                    CNNLayer::Ptr prevLayer = iter->insData[0].lock()->creatorLayer.lock();
                    if (prevLayer && (CaselessEq<std::string>()(prevLayer->type, "convolution") ||
                                      CaselessEq<std::string>()(prevLayer->type, "fullyconnected"))) {
                        iter->insData[0].lock()->setPrecision(Precision::U8);
                    }
                    // if there is a patter A0 -> Eltwise -> ReLU and Convolution -> Eltwise -> ReLU,
                    // need to mark data after conv as U8
                    if (prevLayer && CaselessEq<std::string>()(prevLayer->type, "eltwise")) {
                        iter->insData[0].lock()->setPrecision(Precision::U8);
                        // decising which input will be used for fusion conv-sum-relu
                        CNNLayer::Ptr input1 = prevLayer->insData[0].lock()->creatorLayer.lock();
                        CNNLayer::Ptr input2 = prevLayer->insData[1].lock()->creatorLayer.lock();
                        CNNLayer::Ptr convLayer = nullptr;
                        CNNLayer::Ptr sumLayer = nullptr;

                        if (!CaselessEq<std::string>()(input1->type, "convolution")) {
                            sumLayer = input1;
                            convLayer = input2;
                        } else {
                            // it covers a case when both inputs are convolutions or when first input is not convolution
                            convLayer = input1;
                            sumLayer = input2;
                        }
                        convLayer->outData[0]->setPrecision(sumLayer->outData[0]->getPrecision());
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "pooling")) {
            auto pool = dynamic_cast<PoolingLayer *>(iter.get());
            if (pool == nullptr) {
                THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer '" << iter->name << "' to pooling";
            }

            if (pool->_type == PoolingLayer::MAX ||
                (pool->_type == PoolingLayer::AVG && pool->outData.size() == 1)) {
                auto prevLayer = iter->insData[0].lock()->creatorLayer.lock();
                if (prevLayer && (prevLayer->precision == Precision::I8 || prevLayer->precision == Precision::U8)) {
                    iter->precision = Precision::I8;
                    iter->outData[0]->setPrecision(
                        statHelper.hasNegativeOutput(iter->name) ? Precision::I8 : Precision::U8);
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "concat")) {
            // we can do safe
            // casting to concat and take axis parameter
            // we can concat scales only if concat does concatination by feature maps
            bool axisFeatureMaps = false;
            auto concatLayer = dynamic_cast<ConcatLayer *>(iter.get());
            if (concatLayer) {
                if (concatLayer->_axis == 1
                    && concatLayer->insData.size()
                    && concatLayer->insData[0].lock()->getTensorDesc().getDims().size() == 4) {
                    axisFeatureMaps = true;
                }
            } else {
                THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer " << iter->name << " to concat";
            }

            if (axisFeatureMaps) {
                // verification of input data types
                bool inputFP32 = false;
                bool inputI8 = false;
                bool inputU8 = false;

                for (auto inputData : iter->insData) {
                    auto data = inputData.lock();
                    if (data->getPrecision() == Precision::FP32) {
                        inputFP32 = true;
                    } else if (data->getPrecision() == Precision::I8) {
                        inputI8 = true;
                    } else if (data->getPrecision() == Precision::U8) {
                        inputU8 = true;
                    } else {
                        // Is it a case of input, i.e. passing I16 to concat?
                        // TODO(amalyshe) to handle inputs as a separate usecase
                        THROW_IE_EXCEPTION << "I8 normalizer: input data has unknown precision on the edge for concat: " << data->name;
                    }
                }

                if (inputFP32) {
                    for (auto i : iter->insData) {
                        if (i.lock()->creatorLayer.lock()->precision != Precision::FP32) {
                            returnTailToFP32(i.lock()->creatorLayer.lock());
                        }
                    }
                } else {
                    iter->precision = Precision::I8;

                    // we set outpout precision to U8 only if all inputs are U8, in other case it will be I8
                    auto outputPrecision = (inputU8 && !inputI8) ? Precision::U8 : Precision::I8;

                    // if we have mixed input for I8 and U8, we have to insert scale to edges having U8 to convert to I8
                    // Yes, it leads to loosing of some precision and might lead to some performance degradation
                    // until we have scale supporting s8/u8 input and s8/u8 output.
                    if (inputU8 && inputI8) {
                        // looking for all edges having U8
                        for (size_t d = 0; d < iter->insData.size(); d++) {
                            auto data = iter->insData[d].lock();
                            if (data->getPrecision() == Precision::U8) {
                                size_t c = static_cast<size_t>(data->getDims()[1]);

                                std::vector<float> ssWValues;
                                std::vector<float> ssSValues;
                                for (auto i = 0; i < c; i++) {
                                    ssWValues.push_back(1.0f);
                                    ssSValues.push_back(0.0f);
                                }
                                std::string layerName = data->creatorLayer.lock()->name + "_ScaleShift_U8I8_" + iter->name;
                                CNNLayer::Ptr newLayer = createDWConvolutionForScale(layerName, c, ssWValues.data(), ssSValues.data());
                                newLayer->precision = Precision::I8;
                                AddLayerToCNNNetworkBeforeLayer(newLayer, iter, d);

                                // update statistic to pass quantization smoothly
                                std::string inputLayerName = newLayer->insData[0].lock()->creatorLayer.lock()->name;
                                statHelper.copyStatistics(inputLayerName, layerName);
                                newLayer->outData[0]->setPrecision(Precision::I8);
                            }
                        }
                    }

                    if (iter->outData.size() == 1) {
                        for (auto &&out : iter->outData) {
                            out->setPrecision(outputPrecision);
                        }
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "eltwise")) {
            // we decide which of the layers will be in int-8 mode and initialize special scale which will be used
            // later in "conv-sum-relu" fuse. i8 execution of eltwise always assume this fusion
            if (isNextFusionAllowed(iter)) {
                if (iter->insData.size() == 2) {
                    CNNLayer::Ptr input1 = iter->insData[0].lock()->creatorLayer.lock();
                    CNNLayer::Ptr input2 = iter->insData[1].lock()->creatorLayer.lock();
                    if ((CaselessEq<std::string>()(input1->type, "convolution")
                         || CaselessEq<std::string>()(input2->type, "convolution")) &&
                        !CaselessEq<std::string>()(input1->type, "concat") &&
                        !CaselessEq<std::string>()(input2->type, "concat") &&
                        input1->precision != Precision::FP32 &&
                        input2->precision != Precision::FP32) {
                        // understand which layer will be used for sum
                        CNNLayer::Ptr sumLayer = nullptr;
                        CNNLayer::Ptr convLayer = nullptr;

                        if (!CaselessEq<std::string>()(input1->type, "convolution")) {
                            sumLayer = input1;
                            convLayer = input2;
                        } else {
                            // it covers a case when both inputs are convolutions or when first input is not convolution
                            sumLayer = input2;
                            convLayer = input1;
                        }

                        // mark eltwise as a I8 executable, mark out data as I8
                        iter->precision = Precision::I8;
                        iter->outData[0]->setPrecision(Precision::I8);
                        // calculate the only scale
                        Blob::Ptr sumLayerScales = statHelper.getOutputScale(statHelper.getLatestInFuse(sumLayer));
                        Blob::Ptr convLayerScales = statHelper.getOutputScale(statHelper.getLatestInFuse(convLayer));
                        float *sumScale = sumLayerScales->buffer().as<float *>();
                        float *convScale = convLayerScales->buffer().as<float *>();
                        for (size_t i = 0; i < sumLayerScales->size(); i++) {
                            sumScale[i] /= convScale[i];
                        }

                        iter->blobs["eltwise-sum-scale"] = sumLayerScales;
                    }
                }
            } else {
                // if there are convolutions are inputs to this eltwise, we forcedly move them to FP32
                for (auto i : iter->insData) {
                    auto type = i.lock()->creatorLayer.lock()->type;
                    if (CaselessEq<std::string>()(type, "convolution") ||
                        CaselessEq<std::string>()(type, "fullyconnected")) {
                        i.lock()->creatorLayer.lock()->precision = Precision::FP32;
                        i.lock()->setPrecision(Precision::FP32);
                    }
                }
            }
        } else if (CaselessEq<std::string>()(iter->type, "resample")) {
            iter->precision = Precision::I8;
            iter->outData[0]->setPrecision(iter->insData[0].lock()->getPrecision());
        }
    }

    // quantization of weights/biases
    sortedLayers = CNNNetSortTopologically(net);
    for (auto iter : sortedLayers) {
        if (iter->precision == Precision::I8 &&
                (CaselessEq<std::string>()(iter->type, "convolution") ||
                 CaselessEq<std::string>()(iter->type, "fullyconnected"))) {
            QuantizeConvolutionOrFullyConnected(iter, statHelper);
        }
    }

    // Returning of tails to FP32 mode if optimistic approach marked them as I8
    // no sense to do pooling in i8, we can return just after convolution
    for (auto iter : sortedLayers) {
        // TODO(amalyshe) here is a handling of case when iter provides data to the only one next layer
        // need to extend to cases when it provides data to many layers
        if (iter->precision == Precision::I8
            && iter->outData.size() == 1) {
            if ((iter->outData[0]->inputTo.size() == 1
                && iter->outData[0]->inputTo.begin()->second->precision == Precision::FP32)
                || iter->outData[0]->inputTo.size() == 0) {
                returnTailToFP32(iter);
            }
        }
    }
}

void CNNNetworkInt8Normalizer::PropagateScaleFactors(CNNNetwork& net, const CNNStatisticHelper& statHelper) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    // Moving o-scales down
    for (auto iter : sortedLayers) {
        if (iter->type == "Concat" && iter->precision == Precision::I8) {
            // Checking if all inputs are INT8
            bool all_inputs_are_int8 = true;
            for (int k = 0; k < iter->insData.size(); k++) {
                auto prevKLayer = iter->insData[k].lock()->creatorLayer.lock();
                if ((prevKLayer->precision != Precision::I8 && prevKLayer->precision != Precision::U8) ||
                    prevKLayer->blobs.find("i-concat-scale") == prevKLayer->blobs.end()) {
                    all_inputs_are_int8 = false;
                    break;
                }
            }

            if (all_inputs_are_int8) {
                // Merging o-scales of the inputs to make one for the Concat
                // Creating the o-scale for the Concat by concatenating the input concats
                size_t outputChannels = iter->outData[0]->dims[2];

                std::shared_ptr<Data> oScaleData = std::shared_ptr<Data>(new Data("o-scale", { outputChannels }, Precision::FP32, Layout::C));
                auto oScale = CreateBlobFromData(oScaleData);
                oScale->allocate();

                float* oScaleMemory = static_cast<float*>(oScale->buffer());
                int cc = 0;
                for (int in = 0; in < iter->insData.size(); in++) {
                    auto prevOScale = iter->insData[in].lock()->creatorLayer.lock()->blobs["i-concat-scale"];
                    float* prevOScaleMemory = static_cast<float*>(prevOScale->buffer());

                    for (int c = 0; c < prevOScale->size(); c++) {
                        oScaleMemory[cc] = prevOScaleMemory[c];
                        cc++;
                    }
                }
                if (cc != outputChannels) THROW_IE_EXCEPTION << "Size of o-scale after " << iter->name << " isn't equal to the channels count";

                iter->precision = Precision::I8;
                iter->blobs["o-scale"] = oScale;
            }
        }

        if (iter->blobs.find("o-scale") != iter->blobs.end()) {
            int int8Consumers = 0;
            int fp32Consumers = 0;
            if (iter->outData.size() > 1) {
                THROW_IE_EXCEPTION << "normalization algorithm for int8 found layer having o-scale and multiple ports";
            }
            if (iter->outData.size() == 1) {
                for (auto l : iter->outData[0]->inputTo) {
                    if (l.second->precision == Precision::I8 || l.second->precision == Precision::U8) {
                        if (CaselessEq<std::string>()(l.second->type, "Pooling") ||
                            CaselessEq<std::string>()(l.second->type, "ReLU") ||
                            CNNNetworkInt8Normalizer::isReLULikeClamp(l.second)
                        ) {
                            l.second->blobs["o-scale"] = iter->blobs["o-scale"];
                            // debug scales. Need to compare with actual values in FP32 scoring
                            l.second->blobs["ext-scale"] = l.second->blobs["o-scale"];
                            int8Consumers++;
                        } else if (l.second->type == "Convolution") {
                            l.second->blobs.erase("i-scale");
                            int8Consumers++;
                        } else if (CaselessEq<std::string>()(l.second->type, "Eltwise")) {
                            if (statHelper.getLatestInFuse(iter) != iter) {
                                l.second->blobs["o-scale"] = iter->blobs["o-scale"];
                            }
                            int8Consumers++;
                        } else if ((l.second->precision == Precision::I8 || l.second->precision == Precision::U8) &&
                                   CaselessEq<std::string>()(l.second->type, "Resample")) {
                            // If resample has concat as input layer it should inherit it's
                            // output scale
                            if (l.second->insData.size() == 1) {
                                CNNLayerPtr creator = l.second->insData[0].lock()->creatorLayer.lock();
                                if (CaselessEq<std::string>()(creator->type, "Concat")) {
                                    l.second->blobs["o-scale"] = creator->blobs["o-scale"];
                                    l.second->blobs["i-concat-scale"] = l.second->blobs["o-scale"];
                                }
                            }

                            // No concat found, let use statistics
                            if (l.second->blobs.find("o-scale") == l.second->blobs.end()) {
                                auto oScale = statHelper.getOutputScale(l.second);
                                l.second->blobs["o-scale"] = oScale;
                                l.second->blobs["i-concat-scale"] = l.second->blobs["o-scale"];
                            }
                            int8Consumers++;
                        } else if ((l.second->precision == Precision::I8) &&
                            CaselessEq<std::string>()(l.second->type, "concat")) {
                            // if concat is i8, we can propagate oscale further to concat.
                            // The logic around o-scale assumes that if we have it in the layer after iteration
                            // in this loop it means that it must not be removed and we need to place
                            // scale. While for concat we return to one layer back and again need to analyze o-scale
                            // and it is not clear if we need to return o-scale or it was only for concat.
                            // Having all of this in mind, it's better to rename o-scale to i-concat-scale
                            iter->blobs["i-concat-scale"] = iter->blobs["o-scale"];
                            int8Consumers++;
                        } else {
                            fp32Consumers++;
                        }
                    } else if (CaselessEq<std::string>()(l.second->type, "priorbox") ||
                        CaselessEq<std::string>()(l.second->type, "priorboxclustered")) {
                    } else {
                        // we are leaving o-scale still for adding of scale-shift before FP32 layer
                        fp32Consumers++;
                    }
                }

                if (iter->outData[0]->inputTo.empty()) {
                    fp32Consumers++;
                }

                if (CaselessEq<std::string>()(iter->type, "Convolution") ||
                    CaselessEq<std::string>()(iter->type, "FullyConnected")) {
                    if (int8Consumers) {
                        iter->blobs["oi-scale"] = iter->blobs["o-scale"];
                    } else {
                        iter->outData[0]->setPrecision(Precision::FP32);
                    }
                }
                if (!fp32Consumers) {
                    iter->blobs.erase("o-scale");
                }
            }
        }
    }

    // fixing cornercases when o-scale was propagated through linear tail but it is more efficient to leave
    // conversion to de-normalized values in convolution
    for (auto iter : sortedLayers) {
        if (iter->blobs.find("o-scale") != iter->blobs.end()) {
            // go over out data. if all outputs are fp32, continue this optimization
            bool canOptimize = true;

            // current layer must not be convolution
            if (CaselessEq<std::string>()(iter->type, "convolution")) {
                canOptimize = false;
            }
            for (auto o : iter->outData) {
                for (auto ol : o->inputTo) {
                    if (ol.second->precision == Precision::I8) {
                        canOptimize = false;
                    }
                }
            }
            if (!canOptimize) {
                continue;
            }
            // trying to go up until convolution
            auto curLayer = iter;
            bool eliminateOScale = true;
            while (curLayer
                   && curLayer->blobs.find("oi-scale") == curLayer->blobs.end()
                   && eliminateOScale) {
                if (curLayer->insData.size() == 1
                    && curLayer->insData[0].lock()->creatorLayer.lock()
                    && curLayer->insData[0].lock()->creatorLayer.lock()->outData.size() == 1
                    && curLayer->insData[0].lock()->inputTo.size() == 1) {
                    curLayer = curLayer->insData[0].lock()->creatorLayer.lock();
                    if (!CaselessEq<std::string>()(curLayer->type, "Pooling")
                        && !CaselessEq<std::string>()(curLayer->type, "ReLU")
                        && !isReLULikeClamp(curLayer)
                        && !CaselessEq<std::string>()(curLayer->type, "Convolution")) {
                        eliminateOScale = false;
                    }
                } else {
                    eliminateOScale = false;
                }
            }
            if (eliminateOScale && curLayer) {
                for (auto o : iter->outData) {
                    o->setPrecision(Precision::FP32);
                }
                for (auto o : curLayer->outData) {
                    o->setPrecision(Precision::FP32);
                }

                curLayer->blobs.erase("oi-scale");
                iter->blobs.erase("o-scale");
                auto iLayer = iter;
                while (iLayer != curLayer) {
                    if (iLayer->type == "Pooling") {
                        iLayer->precision = Precision::FP32;
                    }
                    iLayer = iLayer->insData[0].lock()->creatorLayer.lock();
                }
            }
        }
    }
}

void precisionColoring(const CNNLayerPtr layer,
    ordered_properties &printed_properties,
    ordered_properties &node_properties) {
    // looking for the w-scale
    if (layer->blobs.find("w-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("w-scale", ""));
    }

    // looking for the oi-scale
    if (layer->blobs.find("oi-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("oi-scale", ""));
    }

    // looking for the o-scale
    if (layer->blobs.find("o-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("o-scale", ""));
    }
    // looking for the i-scale
    if (layer->blobs.find("i-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("i-scale", ""));
    }

    printed_properties.insert(printed_properties.begin(),
        std::pair<std::string, std::string>("Precision", layer->precision == Precision::FP32 ? "FP32" : "I8"));

    if (layer->precision == Precision::FP32) {
        node_properties.emplace_back("fillcolor", "#5A5DF0");
    } else {
        node_properties.emplace_back("fillcolor", "#20F608");
    }
}

void CNNNetworkInt8Normalizer::NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats) {
    CNNNetwork cnnn(&network);

    int maxSign = 0x7F;
    int maxUnsign = 0xFF;

    // Applying int8-conversion
    StatsMap statsMap = netStats.getNodesStats();

    CNNStatisticHelper statHelper(cnnn, statsMap, maxSign, maxUnsign);

    replaceScaleShiftByDWConvolution(cnnn);

    DefinesExecutionPrecision(cnnn, statHelper);
    PropagateScaleFactors(cnnn, statHelper);
    ClampsToReLU(cnnn, statHelper);
    AddScaleShifts(cnnn, statHelper);
#ifndef NDEBUG
    std::ofstream file("i8_normalized.dot");
    saveGraphToDot(cnnn, file, precisionColoring);
#endif
}
