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
#ifndef NDEBUG
    {
        std::ofstream file("bf16_icnnnetwork.dot");
        saveGraphToDot(network, file, precisionColoringBF16);
    }
#endif
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
