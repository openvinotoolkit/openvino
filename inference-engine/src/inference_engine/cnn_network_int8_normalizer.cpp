// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "cnn_network_int8_normalizer.hpp"

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cassert>
#include <cmath>

#include <ie_common.h>
#include <graph_tools.hpp>
#include <blob_factory.hpp>
#include <data_stats.h>
#include "cnn_network_impl.hpp"
#include "cnn_network_stats_impl.hpp"
#include "debug.h"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

void CNNNetworkInt8Normalizer::AddLayerToCNNNetwork(CNNNetwork& net, CNNLayerPtr firstNode, CNNLayerPtr secondNode, CNNLayerPtr nodeToInsert) {
    std::string layerName = nodeToInsert->name;

    int out_index = -1;
    for (int k = 0; k < firstNode->outData.size(); k++) {
        if (firstNode->outData[k]->inputTo.find(secondNode->name) != firstNode->outData[k]->inputTo.end()) {
            out_index = k;
            break;
        }
    }

    TensorDesc desc(firstNode->outData[out_index]->getTensorDesc());
    DataPtr out = DataPtr(new Data(layerName, desc));
    out->creatorLayer = nodeToInsert;

    for (int i = 0; i < secondNode->insData.size(); i++) {
        if (secondNode->insData[i].lock()->creatorLayer.lock() == firstNode) {
            secondNode->insData[i] = out;
            break;
        }
    }

    out->inputTo[secondNode->name] = secondNode;
    firstNode->outData[out_index]->inputTo.erase(secondNode->name);

    firstNode->outData[out_index]->inputTo[layerName] = nodeToInsert;

    nodeToInsert->insData.push_back(firstNode->outData[0]);
    nodeToInsert->outData.push_back(out);
    ((ICNNNetwork&) net).addLayer(nodeToInsert);
}

void CNNNetworkInt8Normalizer::AddScaleShiftBeforeAndAfterInt8(CNNNetwork& net) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    for (auto iter : sortedLayers) {
        if (iter->precision == Precision::I8) {
            // Checking if we have i-scale in the layer
            if (iter->blobs.find("i-scale") != iter->blobs.end()) {
                for (int pi = 0; pi < iter->insData.size(); pi++) {
                    auto previousNode = iter->insData[pi].lock()->getCreatorLayer().lock();

                    auto iScaleBlob = iter->blobs["i-scale"];
                    if (iScaleBlob != nullptr) {
                        // Adding a ScaleShift after Reorder

                        // Creating a ScaleShiftLayer
                        std::string layerName = iter->name + "_scaleshift_before";
                        LayerParams ssCnnLayerParams{ layerName, "ScaleShift", Precision::FP32 };

                        CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

                        AddLayerToCNNNetwork(net, previousNode, iter, ssCnnLayer);

                        size_t C = static_cast<size_t>(previousNode->outData[0]->getDims()[1]);

                        if (iScaleBlob->dims().size() != 1 || C != iScaleBlob->dims()[0]) {
                            THROW_IE_EXCEPTION<< "i-scale dimension should be equal to channels count for layer " << previousNode->name;
                        }

                        float* iScaleBuffer = static_cast<float*>(iScaleBlob->buffer());
                        {
                            ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());

                            // Setting "scales"
                            SizeVector weightsSize = { C };
                            TensorDesc weightsDesc(Precision::FP32, weightsSize, InferenceEngine::C);
                            scshLayer->_weights = InferenceEngine::make_shared_blob<float>(weightsDesc);
                            scshLayer->_weights->allocate();
                            float * weightsData = scshLayer->_weights->buffer();
                            for (size_t i = 0; i < C; i++) {
                                weightsData[i] = 1.0 / iScaleBuffer[i];
                            }

                            // Setting "shifts"
                            SizeVector shiftsSize = { C };
                            TensorDesc shiftsDesc(Precision::FP32, shiftsSize, InferenceEngine::C);
                            scshLayer->_biases = InferenceEngine::make_shared_blob<float>(shiftsDesc);
                            scshLayer->_biases->allocate();
                            float * biasesData = scshLayer->_biases->buffer();
                            for (size_t i = 0; i < C; i++) {
                                biasesData[i] = 0.f;  // Setting to constant "0"
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto iter : sortedLayers) {
        if (iter->precision == Precision::I8) {
            // Checking if we have i-scale in the layer
            if (iter->blobs.find("o-scale") != iter->blobs.end()) {
                auto nextNode = iter->outData[0]->getInputTo().begin()->second;
                if (nextNode->type != "ReLU") {
                    THROW_IE_EXCEPTION<< "Only Conv-ReLU is supported so far: " << iter->name;
                }

                for (auto next : nextNode->outData[0]->getInputTo()) {
                    auto nextNextNode = next.second;

                    auto oScaleBlob = iter->blobs["o-scale"];
                    if (oScaleBlob != nullptr) {
                        // Creating a ScaleShiftLayer
                        std::string layerName = iter->name + "_scaleshift_after";
                        LayerParams ssCnnLayerParams{ layerName, "ScaleShift", Precision::FP32 };

                        CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

                        AddLayerToCNNNetwork(net, nextNode, nextNextNode, ssCnnLayer);

                        size_t C = static_cast<size_t>(nextNode->outData[0]->getDims()[1]);

                        if (oScaleBlob->dims().size() != 1 || C != oScaleBlob->dims()[0]) {
                            THROW_IE_EXCEPTION<< "o-scale dimension should be equal to channels count for layer " << nextNode->name;
                        }

                        float* oScaleBuffer = static_cast<float*>(oScaleBlob->buffer());
                        {
                            ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());

                            // Setting "scales"
                            SizeVector weightsSize = { C };
                            TensorDesc weightsDesc(Precision::FP32, weightsSize, InferenceEngine::C);
                            scshLayer->_weights = InferenceEngine::make_shared_blob<float>(weightsDesc);
                            scshLayer->_weights->allocate();
                            float * weightsData = scshLayer->_weights->buffer();
                            for (size_t i = 0; i < C; i++) {
                                weightsData[i] = oScaleBuffer[i];
                            }

                            // Setting "shifts"
                            SizeVector shiftsSize = { C };
                            TensorDesc shiftsDesc(Precision::FP32, shiftsSize, InferenceEngine::C);
                            scshLayer->_biases = InferenceEngine::make_shared_blob<float>(shiftsDesc);
                            scshLayer->_biases->allocate();
                            float * biasesData = scshLayer->_biases->buffer();
                            for (size_t i = 0; i < C; i++) {
                                biasesData[i] = 0.f;  // Setting to constant "0"
                            }
                        }
                    }
                }
            }
        }
    }
}

void CNNNetworkInt8Normalizer::ScaleDataToInt8(const float* srcData, size_t srcSize, Blob::Ptr int8blob, float maxValue, const std::vector<float>& scales) {
    if (scales.size() == 0 || /*srcblob->size()*/srcSize % scales.size() != 0) {
        THROW_IE_EXCEPTION<< "Wrong number of scale factors";
    }

    size_t channels = scales.size();
    size_t channelSize = /*srcblob->size()*/srcSize / channels;

    const float* data = srcData;
    int8_t* int8data = static_cast<int8_t*>(int8blob->buffer());

    size_t offset;

    float val;

    for (size_t ch = 0; ch < channels; ch++) {
        offset = channelSize * ch;

        for (size_t i = 0; i < channelSize; i++) {
            val = data[offset + i] * scales[ch];

            if (val > maxValue) {
                val = maxValue;
            } else if (val < -maxValue) {
                val = -maxValue;
            }

            int8data[offset + i] = val;
        }
    }
}

void CNNNetworkInt8Normalizer::ConvertToInt8(int maxSign, int maxUnsign, CNNNetwork& net, const std::map<std::string, NetworkNodeStatsPtr>& netNodesStats) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);

    for (auto iter : sortedLayers) {
        if (netNodesStats.find(iter->name) == netNodesStats.end()) {
            continue;
        }

        if (iter->type == "Convolution" /* TODO || iter->type == "FullyConnected" */) {
            auto weights = iter->blobs["weights"];
            auto biases = iter->blobs["biases"];

            // Creating int8 weights blob
            std::shared_ptr<Data> int8WeightsData = std::shared_ptr<Data>(new Data("weights", weights->dims(), Precision::I8, weights->layout()));
            auto int8weights = CreateBlobFromData(int8WeightsData);
            int8weights->allocate();
            iter->blobs["weights"] = int8weights;

            // Creating int8 biases blob
            std::shared_ptr<Data> int8BiasesData = std::shared_ptr<Data>(new Data("biases", biases->dims(), Precision::I8, biases->layout()));
            auto int8biases = CreateBlobFromData(int8BiasesData);
            int8biases->allocate();
            iter->blobs["biases"] = int8biases;

            std::vector<float> weightScalers;

            size_t inputChannels = iter->insData[0].lock()->dims[2];
            size_t outputChannels = iter->outData[0]->dims[2];

            if (iter->type == "FullyConnected") {
                inputChannels = 1;
            }

            // Creating i-scale blob
            auto previousLayer = iter->insData[0].lock()->creatorLayer.lock();
            std::string inputLayerName = previousLayer->name;
            std::shared_ptr<Data> iScaleData = std::shared_ptr<Data>(new Data("i-scale", { inputChannels }, Precision::FP32, Layout::C));
            auto iScale = CreateBlobFromData(iScaleData);
            iScale->allocate();
            float* iScaleMemory = static_cast<float*>(iScale->buffer());
            bool previousFound = false;

            if (!previousFound) {
                for (int c = 0; c < inputChannels; c++) {
                    std::cout << "input max[c]: " << netNodesStats.at(inputLayerName)->_maxOutputs[c] << std::endl;

                    iScaleMemory[c] = fmax(fabs(netNodesStats.at(inputLayerName)->_maxOutputs[c]),
                                           fabs(netNodesStats.at(inputLayerName)->_minOutputs[c]))
                                      / static_cast<float>(maxUnsign);

                    std::cout << "i-scale[c]: " << iScaleMemory[c] << std::endl;
                }
            }
            iter->blobs["i-scale"] = iScale;

            // Creating w-scale blob

            const float* weight = static_cast<const float*>(weights->buffer());
            const float* bias = static_cast<const float*>(biases->buffer());

            std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale
            {
                size_t W_CO = outputChannels,
                        W_CI = inputChannels,
                        W_HW = weights->dims()[0] / inputChannels / outputChannels;

                for (size_t co = 0; co < W_CO; co++) {
                    for (size_t ci = 0; ci < W_CI; ci++) {
                        for (size_t hw = 0; hw < W_HW; hw++) {
                            newWeights.push_back(weight[co * W_CI * W_HW + ci * W_HW + hw] * iScaleMemory[ci]);
                        }
                    }
                }
            }

            size_t outChannelSize = weights->dims()[0] / outputChannels;

            // Calculating weights normalization scale factor (w-scale)
            float* weight_iter;
            size_t co;
            for (co = 0, weight_iter = &newWeights[0]; co < outputChannels; co++, weight_iter += outChannelSize) {
                float max = FLT_MIN;
                DataStats::GetDataAbsMax(weight_iter, outChannelSize, max);

                float scaler = maxSign / max;
                std::cout << "scaler: " << scaler << std::endl;
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
            std::shared_ptr<Data> oScaleData = std::shared_ptr<Data>(new Data("o-scale", { outputChannels }, Precision::FP32, Layout::C));
            auto oScale = CreateBlobFromData(oScaleData);
            oScale->allocate();

            float* oScaleMemory = static_cast<float*>(oScale->buffer());
            for (int c = 0; c < outputChannels; c++) {
                std::cout << "output max[c]: " << netNodesStats.at(iter->name)->_maxOutputs[c] << std::endl;

                oScaleMemory[c] = fmax(fabs(netNodesStats.at(iter->name)->_maxOutputs[c]),
                                       fabs(netNodesStats.at(iter->name)->_minOutputs[c]))
                                  / static_cast<float>(maxSign);

                std::cout << "o-scale[c]: " << oScaleMemory[c] << std::endl;
            }
            iter->blobs["o-scale"] = oScale;

            // Normalizing the weights and the biases
            ScaleDataToInt8(&newWeights[0], weights->size(), int8weights, maxSign, weightScalers);
            ScaleDataToInt8(bias, biases->size(), int8biases, maxSign, weightScalers);

            // Setting precisions
            iter->precision = Precision::I8;

            for (auto&& in : iter->insData) {
                in.lock()->precision = Precision::U8;
            }
            for (auto&& out : iter->outData) {
                out->precision = Precision::I8;
            }

            std::cout << "Layer " << iter->name << " converted to INT8" << std::endl;
        }
    }
}

void CNNNetworkInt8Normalizer::NormalizeNetwork(ICNNNetwork& network, ICNNNetworkStats& netStats) {
    CNNNetwork cnnn(&network);

    int maxSign = 0x7F;
    int maxUnsign = 0xFF;

    // Applying int8-conversion
    ConvertToInt8(maxSign, maxUnsign, cnnn, dynamic_cast<const CNNNetworkStatsImpl&>(netStats).getNodesStats());

    // Adding ScaleShift layers before and after each Convolution-Activation pair
    AddScaleShiftBeforeAndAfterInt8(cnnn);
}
