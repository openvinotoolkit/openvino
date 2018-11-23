// Copyright (C) 2018 Intel Corporation
//
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

void CNNNetworkInt8Normalizer::AddLayerToCNNNetworkBeforeLayer(CNNLayer::Ptr newLayer, CNNLayer::Ptr successor) {
    // verify if data exists
    if (newLayer && successor && successor->insData.size() == 1) {
        // get the insData
        DataPtr pData = successor->insData[0].lock();

        Data *edge2 = new Data(*pData.get());
        DataPtr newEdge(edge2);
        newEdge->getInputTo().clear();
        newEdge->getInputTo()[successor->name] = successor;
        newEdge->name = newLayer->name;
        newEdge->getCreatorLayer() = newLayer;
        successor->insData[0] = newEdge;
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
        newEdgeAfterLayer->precision = Precision::FP32;

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

void CNNNetworkInt8Normalizer::AddScaleShiftBetween(CNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2) {
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
            fillInScaleShift(scshLayer, c, oScaleBuffer, iScaleBuffer);
        }

        ssCnnLayer->outData[0]->precision = ssCnnLayer->outData[0]->inputTo.begin()->second->precision;
    }
}

void CNNNetworkInt8Normalizer::AddScaleShifts(CNNNetwork& net) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>> pairs;

    for (auto iter : sortedLayers) {
        for (int l1_out_i = 0; l1_out_i < iter->outData.size(); l1_out_i++) {
            for (auto nextIter : iter->outData[l1_out_i]->inputTo) {
                CNNLayer::Ptr next = nextIter.second;

                // Checking for an INT8 convolution with FP32 output
                if (iter->type == "Convolution" && iter->precision == Precision::I8 && next->precision == Precision::FP32) {
                    // Do nothing here
                    // MKLDNNPlugin will generate u8->f32 convolution
                } else if ((iter->precision != Precision::FP32 && next->precision == Precision::FP32) ||
                           (iter->precision == Precision::FP32 && next->precision != Precision::FP32)) {
                    pairs.push_back(std::pair<CNNLayerPtr, CNNLayerPtr>(iter, next));
                }
            }
        }
    }

    for (auto& pair : pairs) {
        AddScaleShiftBetween(net, pair.first, pair.second);
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

NetworkNodeStatsPtr CNNNetworkInt8Normalizer::mergeNetworkNodesStats(std::vector<NetworkNodeStatsPtr> stats) {
    int c = stats[0]->_maxOutputs.size();
    for (auto s : stats) {
        if (s->_maxOutputs.size() != c || s->_minOutputs.size() != c) {
            THROW_IE_EXCEPTION << "Inconsistent stats";
        }
    }

    NetworkNodeStatsPtr res = NetworkNodeStatsPtr(new NetworkNodeStats(c));
    for (int i = 0; i < c; i++) {
        float globalMin = stats[0]->_minOutputs[i], globalMax = stats[0]->_maxOutputs[i];
        for (auto s : stats) {
            if (s->_maxOutputs[i] > globalMax) globalMax = s->_maxOutputs[i];
            if (s->_minOutputs[i] < globalMin) globalMin = s->_minOutputs[i];
        }
        res->_minOutputs[i] = globalMin;
        res->_maxOutputs[i] = globalMax;
    }

    return res;
}

InferenceEngine::Blob::Ptr CNNNetworkInt8Normalizer::calculateScaleFactor(const std::string& name, size_t channels,
                                                                          std::vector<NetworkNodeStatsPtr> stats, int maxInt) {
    for (int k = 0; k < stats.size(); k++) {
        if (stats[k]->_minOutputs.size() != channels || stats[k]->_maxOutputs.size() != channels) {
            THROW_IE_EXCEPTION << "min and max sizes should be equal to channels count";
        }
    }

    // Creating i-scale blob
    std::shared_ptr<Data> iScaleData = std::shared_ptr<Data>(new Data(name, { channels }, Precision::FP32, Layout::C));
    auto iScale = CreateBlobFromData(iScaleData);
    iScale->allocate();
    float* iScaleMemory = static_cast<float*>(iScale->buffer());

    for (int c = 0; c < channels; c++) {
        float maxc = 0;
        for (int k = 0; k < stats.size(); k++) {
            // maxc = fmax(maxc, fabs(stats[k]->_minOutputs[c]));        // TODO Check if we should take minimums into account
            maxc = fmax(maxc, fabs(stats[k]->_maxOutputs[c]));
        }

        iScaleMemory[c] = maxc / static_cast<float>(maxInt);

        if (fabs(iScaleMemory[c]) < 1e-7) {
            iScaleMemory[c] = 1.0f;
        }
    }
    return iScale;
}

std::vector<NetworkNodeStatsPtr> splitStats(NetworkNodeStatsPtr stats, std::vector<size_t> channels) {
    NetworkNodeStats s = *stats.get();  // Copying the stats
    std::vector<NetworkNodeStatsPtr> res;

    size_t j = 0;
    for (size_t ci = 0; ci < channels.size(); ci++) {
        NetworkNodeStatsPtr latest = NetworkNodeStatsPtr(new NetworkNodeStats(channels[ci]));
        for (size_t k = 0; k < channels[ci]; k++) {
            if (j > stats->_minOutputs.size()) THROW_IE_EXCEPTION << "Incorrect stats or channels";
            latest->_minOutputs[k] = stats->_minOutputs[j];
            latest->_maxOutputs[k] = stats->_maxOutputs[j];
            j++;
        }
        res.push_back(latest);
    }
    return res;
}

void CNNNetworkInt8Normalizer::ConvertToInt8(int maxSign, int maxUnsign, CNNNetwork& net, const std::map<std::string, NetworkNodeStatsPtr>& netNodesStats) {
    std::map<std::string, NetworkNodeStatsPtr> internalNodesStats = netNodesStats;
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    // Back iterating the network, searching for the "Eltwise-driven subnets"
    // Eltwise-driven subnet is a subnet which ends with an Eltwise and starts with I8-convolution.
    // All the nodes in the subnet should have the same o-scales
    std::vector<CNNLayerPtr> backSortedLayers = sortedLayers;
    std::reverse(std::begin(backSortedLayers), std::end(backSortedLayers));

    std::set<CNNLayerPtr> skippedEltwises;
    // Back propagating statistics
    std::set<CNNLayerPtr> eltwisesProcessed;
    for (auto iter : backSortedLayers) {
        if (iter->params.find("quantization_level") != iter->params.end() && iter->params["quantization_level"] == "FP32") {
            continue;
        }
        if (internalNodesStats.find(iter->name) == internalNodesStats.end()) {
            continue;
        }

        if (iter->type == "Eltwise") {
            // Counting Eltwises in a row
            std::set<CNNLayerPtr> eltwisesSequence;
            CNNLayerPtr ptr = iter;
            bool added;
            do {
                added = false;
                for (auto& n : ptr->insData) {
                    CNNLayerPtr in = n.lock()->creatorLayer.lock();
                    if (in->type == "ReLU") {
                        in = in->insData[0].lock()->creatorLayer.lock();
                    }

                    if (in->type == "Eltwise") {
                        ptr = in;
                        eltwisesSequence.insert(in);
                        added = true;
                    }
                }
            } while (added);

            if (eltwisesSequence.size() > 5) {
                skippedEltwises.insert(eltwisesSequence.begin(), eltwisesSequence.end());
            }
        }

        if (iter->type == "Eltwise" &&
                eltwisesProcessed.find(iter) == eltwisesProcessed.end() &&
                internalNodesStats.find(iter->name) != internalNodesStats.end()) {
            eltwisesProcessed.insert(iter);

            // Collecting all the convolutions that starts the "Eltwise-driven subnet"
            std::set<DataPtr> edgesToWatch;
            std::map<DataPtr, NetworkNodeStatsPtr> edgeStats;

            for (auto in : iter->insData) {
                edgesToWatch.insert(in.lock());
            }

            // Add the statistics of the Eltwise to each edge
            for (auto e : edgesToWatch) {
                NetworkNodeStatsPtr p = internalNodesStats.find(iter->name)->second;
                edgeStats.insert({ e, p });
            }

            do {
                std::set<DataPtr> previousETW = edgesToWatch;
                // For each LayerToWatch processing all its direct inputs
                for (auto e : previousETW) {
                    auto prevLayer = e->creatorLayer.lock();
                        if (internalNodesStats.find(prevLayer->name) != internalNodesStats.end()) {
                            if (prevLayer->type == "Convolution") {
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = edgeStats[e];
#ifndef NDEBUG
                            std::cout << "Propagated stats from " << e->name << " to " << prevLayer->name
                                    << "(" << internalNodesStats[prevLayer->name]->_maxOutputs[0] << ")" << std::endl;
#endif
                            } else if (prevLayer->type == "Eltwise") {
                                eltwisesProcessed.insert(prevLayer);
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = edgeStats[e];
#ifndef NDEBUG
                            std::cout << "Propagated stats from " << e->name << " to " << prevLayer->name
                                    << "(" << internalNodesStats[prevLayer->name]->_maxOutputs[0] << ")" << std::endl;
#endif
                            for (auto ee : prevLayer->insData) {
                                // Adding the edges before the node to the watch list
                                edgesToWatch.insert(ee.lock());
                                // Propagating the stats upwards
                                edgeStats.insert({ ee.lock(), internalNodesStats[prevLayer->name] });
                            }
                            } else if (prevLayer->type == "Pooling") {
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = edgeStats[e];
#ifndef NDEBUG
                            std::cout << "Propagated stats from " << e->name << " to " << prevLayer->name
                                    << "(" << internalNodesStats[prevLayer->name]->_maxOutputs[0] << ")" << std::endl;
#endif
                            for (auto ee : prevLayer->insData) {
                                // Adding the edges beforereleases/openvino-2018-r4 the node to the watch list
                                edgesToWatch.insert(ee.lock());
                                // Propagating the stats upwards
                                edgeStats.insert({ ee.lock(), internalNodesStats[prevLayer->name] });
                            }
                        } else if (prevLayer->type == "ReLU") {
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = NetworkNodeStatsPtr(new NetworkNodeStats(*edgeStats[e].get()));
                            for (auto& mo : internalNodesStats[prevLayer->name]->_minOutputs) {
                                mo = 0;
                        }
#ifndef NDEBUG
                            std::cout << "Propagated stats from " << e->name << " to " << prevLayer->name
                                    << ", zeroing the minimal values" << "(" << internalNodesStats[prevLayer->name]->_maxOutputs[0] << ")" << std::endl;
#endif
                            for (auto ee : prevLayer->insData) {
                                // Adding the edges before the node to the watch list
                                edgesToWatch.insert(ee.lock());
                                // Propagating the stats upwards
                                edgeStats.insert({ ee.lock(), internalNodesStats[prevLayer->name] });
                    }
                        } else if (prevLayer->type == "Concat") {
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = edgeStats[e];
#ifndef NDEBUG
                            std::cout << "Propagated stats from " << e->name << " to " << prevLayer->name << "("
                                    << internalNodesStats[prevLayer->name]->_maxOutputs[0] << ")" << std::endl;
#endif
                            // Getting the inputs channels counts
                            std::vector<size_t> inputsChannels;
                            for (auto i : prevLayer->insData) {
                                size_t channels = i.lock()->getTensorDesc().getDims()[1];
                                inputsChannels.push_back(channels);
                            }

                            // Splitting the stats to feed them upwards the Concat inputs
                            std::vector<NetworkNodeStatsPtr> inStats = splitStats(internalNodesStats[prevLayer->name], inputsChannels);
                            auto in = prevLayer->insData.begin();
                            for (size_t i = 0; i < inStats.size(); i++) {
                                edgeStats.insert({ in->lock(), inStats[i] });
                                // Adding the edges before the node to the watch list
                                edgesToWatch.insert(in->lock());
                                in++;
                            }

                        } else {
                            // Setting the current node's stats to the stats saved for the edge after it
                            internalNodesStats[prevLayer->name] = edgeStats[e];
                            for (auto ee : prevLayer->insData) {
                                // Adding the edges before the node to the watch list
                                edgesToWatch.insert(ee.lock());
                                // Propagating the stats upwards
                                edgeStats.insert({ ee.lock(), internalNodesStats[prevLayer->name] });
                    }
                }
            }

                    edgesToWatch.erase(e);
                }
            } while (!edgesToWatch.empty());
        }
    }

    // Converting layers to Int8. Calculating the multipliers if needed
    for (auto iter : sortedLayers) {
        if (iter->params.find("quantization_level") != iter->params.end() && iter->params["quantization_level"] == "FP32") {
            continue;
        }
        if (internalNodesStats.find(iter->name) == internalNodesStats.end()) {
            continue;
        }

        if (iter->type == "Eltwise") {
            if (skippedEltwises.find(iter) != skippedEltwises.end()) {
#ifndef NDEBUG
                std::cout << "Skipping Eltwise " << iter->name << " conversion" << std::endl;
#endif
                continue;
            }

            auto eltw = dynamic_cast<EltwiseLayer*>(iter.get());
            if (eltw == nullptr) THROW_IE_EXCEPTION << "Can't interpret " << iter->name << " as an Eltwise layer";

            // Checking if all the previous layers are I8
            bool canConvert = true;
            for (auto in : iter->insData) {
                auto previousLayer = in.lock()->creatorLayer.lock();
                if (previousLayer->precision != Precision::I8) {
                    // If the precision isn't I8, we don't convert the Eltwise
                    canConvert = false;
                }
            }

            if (canConvert && eltw->_operation == EltwiseLayer::eOperation::Sum) {
                // Mark it I8
                iter->precision = Precision::I8;
                if (iter->outData[0]->inputTo.size() == 1 &&
                    iter->outData[0]->inputTo.begin()->second->type == "ReLU") {
                    auto reluLayer = iter->outData[0]->inputTo.begin()->second;

                    // Signed int8 between Eltwise and ReLU
                    for (auto&& out : iter->outData) {
                        out->precision = Precision::I8;
                    }

                    // ReLU after Eltwise is being set to signed int8 type unlike ReLU after a Convolution.
                    // This is the best way to support Eltwise-ReLU-Eltwise chain (that is common in ResNet-like nets)
                    reluLayer->precision = Precision::I8;

                    // Signed int8 after ReLU
                    for (auto&& out : reluLayer->outData) {
                        out->precision = Precision::I8;
                    }
                }
            }
        } else if (iter->type == "Convolution") {
            size_t inputChannels = iter->insData[0].lock()->dims[2];
            size_t outputChannels = iter->outData[0]->dims[2];

            auto previousLayer = iter->insData[0].lock()->creatorLayer.lock();
            std::string inputLayerName = previousLayer->name;

            // for case when we have the only average pooling before, we need to take this
            // statistic from input of avg pooloing to compensate work of average pooling
            // and to stay in int8 as much as we can
            if (previousLayer->type == "Pooling" && (previousLayer->precision == Precision::I8 || previousLayer->precision == Precision::U8)) {
                // take input name to the pooling
                inputLayerName = previousLayer->insData[0].lock()->creatorLayer.lock()->name;
            }


            if (internalNodesStats.find(inputLayerName) == internalNodesStats.end()) {
                THROW_IE_EXCEPTION << "No stats for layer " << inputLayerName;
            }

            // Checking the topology
            if (iter->outData.size() != 1) {
                THROW_IE_EXCEPTION << "Strange convolution with multiple outputs";
            }

            // Checking if we have negative inputs
            float min_inp = 0;
            for (int c = 0; c < inputChannels; c++) {
                if (internalNodesStats.at(inputLayerName)->_minOutputs[c] < min_inp)
                    min_inp = internalNodesStats.at(inputLayerName)->_minOutputs[c];
            }
            // Layer has negative input and can't be converted to INT8
            if (min_inp < 0) {
                continue;
            }

            auto iScale = calculateScaleFactor("i-scale", inputChannels, { internalNodesStats.at(inputLayerName) }, maxUnsign);
            iter->blobs["i-scale"] = iScale;

            Blob::Ptr weights = nullptr;
            Blob::Ptr biases = nullptr;

            Blob::Ptr int8weights = nullptr;
            Blob::Ptr int32biases = nullptr;

            if (iter->blobs.find("weights") != iter->blobs.end()) {
                weights = iter->blobs["weights"];

                // Creating int8 weights blob
                std::shared_ptr<Data> int8WeightsData = std::shared_ptr<Data>(new Data("weights", weights->dims(), Precision::I8, weights->layout()));
                int8weights = CreateBlobFromData(int8WeightsData);
                int8weights->allocate();
                iter->blobs["weights"] = int8weights;
            }

            if (iter->blobs.find("biases") != iter->blobs.end()) {
                biases = iter->blobs["biases"];

                // Creating int8 biases blob
                std::shared_ptr<Data> int32BiasesData = std::shared_ptr<Data>(new Data("biases", biases->dims(), Precision::I32, biases->layout()));
                int32biases = CreateBlobFromData(int32BiasesData);
                int32biases->allocate();
                iter->blobs["biases"] = int32biases;
            }

            std::vector<float> weightScalers;


            // Creating w-scale blob
            if (weights) {
                const float* weight = static_cast<const float*>(weights->buffer());

                ConvolutionLayer* pConv = dynamic_cast<ConvolutionLayer*>(iter.get());
                if (pConv->_group == 0) {
                    THROW_IE_EXCEPTION << "Convolution '" << iter->name << "'has wrong groups number == 0";
                }

                std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale

                size_t W_CO = outputChannels / pConv->_group,
                        W_CI = inputChannels / pConv->_group,
                        W_HW = weights->dims()[0] / W_CI / W_CO / pConv->_group;

                {
                    float* iScaleMemory = static_cast<float*>(iScale->buffer());
                    for (size_t g = 0; g < pConv->_group; g++) {
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
                size_t outChannelSize = weights->dims()[0] / W_CO / pConv->_group;

                // Calculating weights normalization scale factor (w-scale)
                float* weight_iter;
                size_t co;
                for (co = 0, weight_iter = &newWeights[0]; co < outputChannels; co++, weight_iter += outChannelSize) {
                    float max = FLT_MIN;
                    DataStats::GetDataAbsMax(weight_iter, outChannelSize, max);

                    float scaler = static_cast<float>(maxSign) / max;
                    weightScalers.push_back(scaler);
                }

                std::shared_ptr<Data> wScaleData = std::shared_ptr<Data>(new Data("w-scale", { outputChannels }, Precision::FP32, Layout::C));
                auto wScale = CreateBlobFromData(wScaleData);
                wScale->allocate();

                float* wScaleMemory = static_cast<float*>(wScale->buffer());

                for (size_t i = 0; i < outputChannels; i++) {
                    wScaleMemory[i] = 1.0 / weightScalers[i];
                }
                iter->blobs["w-scale"] = wScale;

                // Creating o-scale blob
                // verify if there is ReLU just after the convolution, in this case
                // we will normalize only positive values to maxUnsign
                // anther decision - we will not propagate o-scale for cases if we do not have
                // conv-relu pattern because there is no sense right now to normalize to sihned I8
                // no primitives can process such input for a while
                if (iter->outData[0]->inputTo.size() == 1 &&
                    iter->outData[0]->inputTo.begin()->second->type == "ReLU") {
                    auto reluLayer = iter->outData[0]->inputTo.begin()->second;

                    auto oScale = calculateScaleFactor("o-scale", outputChannels, { internalNodesStats.at(reluLayer->name) }, maxUnsign);
                    iter->blobs["o-scale"] = oScale;

                    // Unsigned int8 precision for ReLU
                    reluLayer->precision = Precision::U8;

                } else {
                    auto oScale = calculateScaleFactor("o-scale", outputChannels, { internalNodesStats.at(iter->name) }, maxUnsign);
                    iter->blobs["o-scale"] = oScale;
                }

                iter->precision = Precision::I8;
                // Normalizing the weights
                ScaleDataToInt(&newWeights[0], weights->size(), int8weights, weightScalers);
            }

            // Normalizing the biases
            if (biases) {
                const float* bias = static_cast<const float*>(biases->buffer());
                ScaleDataToInt(bias, biases->size(), int32biases, weightScalers);
            }
        } else if (iter->type == "Pooling") {
            auto pool = dynamic_cast<PoolingLayer*>(iter.get());
            if (pool && (pool->_type == PoolingLayer::MAX
                || (pool->_type == PoolingLayer::AVG
                    && pool->outData.size() == 1
                    && pool->outData[0]->inputTo.size() == 1
                    && pool->outData[0]->inputTo.begin()->second->type == "Convolution"))) {
                auto prevLayer = iter->insData[0].lock()->creatorLayer.lock();
                if (prevLayer && (prevLayer->precision == Precision::I8 || prevLayer->precision == Precision::U8)) {
                    iter->precision = Precision::I8;
                    if (iter->outData.size() == 1) {
                        for (auto&& out : iter->outData) {
                            out->precision = Precision::U8;
                        }
                    }
                }
            }
        } else if (iter->type == "Concat") {
            bool allParentsInt = true;

#ifndef NDEBUG
            Precision p = iter->insData[0].lock()->precision;
            for (auto inputData : iter->insData) {
                if (inputData.lock()->precision != p) {
                    std::cerr << "WARNING: We have a Concat " << iter->name << " whose inputs have different precisions" << std::endl;
                }
            }
#endif

            for (auto inputData : iter->insData) {
                auto inPrecision = inputData.lock()->creatorLayer.lock()->precision;
                if (inPrecision != Precision::I8 && inPrecision != Precision::U8) {
                    allParentsInt = false;
                }
            }
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
            if (allParentsInt && axisFeatureMaps) {
                iter->precision = Precision::I8;
                if (iter->outData.size() == 1) {
                    for (auto&& out : iter->outData) {
                        out->precision = Precision::U8;
                    }
                }
            } else {
                for (auto&& id : iter->insData) {
                    id.lock()->precision = Precision::FP32;
                }
            }
        }
    }

    // Processing edges precisions
    for (auto iter : sortedLayers) {
       if (iter->params.find("quantization_level") != iter->params.end() && iter->params["quantization_level"] == "FP32") {
           continue;
       }
       if (internalNodesStats.find(iter->name) == internalNodesStats.end()) {
           continue;
       }

       if (iter->type == "Convolution") {
           if (iter->outData[0]->inputTo.size() > 0) {
               auto nextFirstLayer = iter->outData[0]->inputTo.begin()->second;

               // If we have only a single ReLU after the convolution
               if (iter->outData[0]->inputTo.size() == 1 && nextFirstLayer->type == "ReLU") {
                   // Setting precision I8 between the convolution and ReLU
                   // (this will be eliminated by the MKLDNNPlugin GraphOptimizer, but it's beautiful)
                   iter->outData[0]->precision = Precision::I8;
                   // If any integer output found, setting ReLU output to U8
                   nextFirstLayer->outData[0]->precision = Precision::U8;

               } else {
                   // If there is no ReLU after the convolution...
                   for (auto&& inTo : iter->outData[0]->inputTo) {
                       if (inTo.second->precision == Precision::I8 || inTo.second->precision == Precision::U8) {
                           // If any integer output found, setting the convolution output to I8
                           iter->outData[0]->precision = Precision::I8;
                           break;
                       }
                   }
               }
           }
        } else if (iter->type == "Eltwise") {
            if (iter->precision == Precision::I8) {
                size_t outputChannels = iter->outData[0]->dims[2];

                std::vector<NetworkNodeStatsPtr> stats;
                stats.push_back(internalNodesStats.at(iter->name));

                auto oScale = calculateScaleFactor("o-scale", outputChannels, stats, maxUnsign);

                size_t inputChannels = iter->insData[0].lock()->dims[2];

                for (auto inputData : iter->insData) {
                    auto prevData = inputData.lock();
                    auto prevLayer = prevData->creatorLayer.lock();
                    prevData->precision = Precision::I8;
                }

                // Setting the self oScale to the same as the previous convolutions
                iter->blobs["o-scale"] = oScale;
                iter->precision = Precision::I8;

                for (auto&& out : iter->outData) {
                    out->precision = Precision::I8;
                }
            }
        }
    }
}

void CNNNetworkInt8Normalizer::PropagateScaleFactors(CNNNetwork& net) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    std::vector<CNNLayer::Ptr> oScaleLayers;

    // Moving o-scales down
    for (auto iter : sortedLayers) {
        if (iter->type == "Concat" && iter->precision == Precision::I8) {
            // Checking if all inputs are INT8
            bool all_inputs_are_int8 = true;
            for (int k = 0; k < iter->insData.size(); k++) {
                auto prevKLayer = iter->insData[k].lock()->creatorLayer.lock();
                if ((prevKLayer->precision != Precision::I8 && prevKLayer->precision != Precision::U8) ||
                    prevKLayer->blobs.find("o-scale") == prevKLayer->blobs.end()) {
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
                    auto prevOScale = iter->insData[in].lock()->creatorLayer.lock()->blobs["o-scale"];
                    float* prevOScaleMemory = static_cast<float*>(prevOScale->buffer());

                    for (int c = 0; c < prevOScale->size(); c++) {
                        oScaleMemory[cc] = prevOScaleMemory[c];
                        cc++;
                    }
                    iter->insData[in].lock()->creatorLayer.lock()->blobs.erase("o-scale");
                }
                if (cc != outputChannels) THROW_IE_EXCEPTION << "Size of o-scale after " << iter->name << " isn't equal to the channels count";

                iter->precision = Precision::I8;
                iter->blobs["o-scale"] = oScale;
            }
        }

        if (iter->blobs.find("o-scale") != iter->blobs.end()) {
            bool canPropagate = true;
            if (iter->outData.size() > 1) {
                THROW_IE_EXCEPTION << "normalization algorithm for int8 found layer having o-scale and multiple ports";
            }
            if (iter->outData.size() == 1) {
                for (auto l : iter->outData[0]->inputTo) {
                    if (l.second->precision == Precision::I8 || l.second->precision == Precision::U8) {
                        if (l.second->type == "Pooling" || l.second->type == "ReLU") {
                            l.second->blobs["o-scale"] = iter->blobs["o-scale"];
                        } else if (l.second->type == "Convolution") {
                            l.second->blobs.erase("i-scale");
                        } else if (l.second->type == "Eltwise") {
                            canPropagate = true;
                        } else {
                            canPropagate = false;
                        }
                    } else {
                        // we are leaving o-scale still for adding of scale-shift before FP32 layer
                        canPropagate = false;
                    }
                }

                if (iter->outData[0]->inputTo.empty()) {
                    canPropagate = false;
                }

                if (canPropagate) {
                    if (iter->type == "Convolution") {
                        iter->blobs["oi-scale"] = iter->blobs["o-scale"];
                    }
                    iter->blobs.erase("o-scale");
                } else {
                    if (iter->type == "Convolution") {
                        iter->blobs.erase("o-scale");
                    }
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
                    if (curLayer->type != "Pooling"
                        && curLayer->type != "ReLU"
                        && curLayer->type != "Convolution") {
                        eliminateOScale = false;
                    }
                } else {
                    eliminateOScale = false;
                }
            }
            if (eliminateOScale && curLayer) {
                for (auto o : iter->outData) {
                    o->precision = Precision::FP32;
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

StatsMap ConvertAllStatsToMax(const ICNNNetwork &network, const StatsMap &statsMap) {
    StatsMap newMap = statsMap;

    float dummy;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (auto l : sortedLayers) {
        auto it = newMap.find(l->name);
        if (l->type == "Pooling") {
            // get predecessor statistic and update it for current layer
            auto parent = l->insData[0].lock()->creatorLayer.lock();
            auto itPStat = newMap.find(parent->name);
            if (itPStat != newMap.end()) {
                newMap[l->name] = itPStat->second;
            } else if (it != newMap.end()) {
                THROW_IE_EXCEPTION << "pool has statistic but parent does not have it. Not implemented case.";
            }
        } else if (it != newMap.end()) {
            float min = FLT_MAX;
            float max = FLT_MIN;

            if (l->type == "Concat"
                && l->outData.size() == 1 && l->outData[0]->getTensorDesc().getDims().size() == 4) {
                size_t concatLayerIdx = 0;
                for (int k = 0; k < l->insData.size(); k++) {
                    auto prevKLayer = l->insData[k].lock()->creatorLayer.lock();
                    // looking for the statistic for prevKLayer
                    auto kLayerStat = newMap.find(prevKLayer->name);
                    if (kLayerStat != newMap.end()) {
                        for (size_t ikStat = 0; ikStat < kLayerStat->second->_maxOutputs.size(); ikStat++, concatLayerIdx++) {
                            it->second->_maxOutputs[concatLayerIdx] = kLayerStat->second->_maxOutputs[ikStat];
                            it->second->_minOutputs[concatLayerIdx] = kLayerStat->second->_minOutputs[ikStat];
                        }
                    } else {
                        THROW_IE_EXCEPTION << "We have incomplete statistic for predecessors of concat layer " << l->name;
                    }
                }
            } else {
                if (!it->second->_maxOutputs.empty()) {
                    DataStats::GetDataAbsMax(&it->second->_maxOutputs[0], it->second->_maxOutputs.size(), max);
                    std::fill(it->second->_maxOutputs.begin(), it->second->_maxOutputs.end(), max);
                }
                if (!it->second->_minOutputs.empty()) {
                    DataStats::GetDataMinMax(&it->second->_minOutputs[0], it->second->_minOutputs.size(), min, dummy);
                    std::fill(it->second->_minOutputs.begin(), it->second->_minOutputs.end(), min);
                }
            }
        }
    }

    return newMap;
}

void CNNNetworkInt8Normalizer::NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats) {
    CNNNetwork cnnn(&network);

    int maxSign = 0x7F;
    int maxUnsign = 0xFF;

    // Applying int8-conversion
    StatsMap statsMap = netStats.getNodesStats();
    statsMap = ConvertAllStatsToMax(network, statsMap);

    ConvertToInt8(maxSign, maxUnsign, cnnn, statsMap);
    PropagateScaleFactors(cnnn);
    AddScaleShifts(cnnn);
#ifndef NDEBUG
    std::ofstream file("i8_normalized.dot");
    saveGraphToDot(cnnn, file, precisionColoring);
#endif
}
