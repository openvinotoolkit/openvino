// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_ir_reader.hpp>
#include "tests_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <transform/transformations/conv_bias_fusion.hpp>
#include <transform/transformations/matmul_bias_fusion.hpp>
#include <transform/transformations/quantizeconv_dequantize_fusion.hpp>
#include <transform/transformations/convert_quantize_conv_elimination.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

using namespace testing;
using namespace InferenceEngine;

class NGraphReaderTests : public TestsCommon {
protected:
    void TearDown() override {}
    void SetUp() override {}

    void compareICNNNetworks(ICNNNetwork::Ptr newNetwork, const CNNNetwork& oldNetwork) {
        auto compareParamVal = [](const std::string& val1, const std::string& val2) -> bool {
            std::vector<std::string> vals1, vals2;
            std::stringstream ss1(val1);
            std::string field;
            while (getline(ss1, field, ',' )) {
                std::stringstream fs(field);
                std::string value;
                fs >> value;
                vals1.emplace_back(value);
            }

            std::stringstream ss2(val2);
            while (getline(ss2, field, ',' )) {
                std::stringstream fs(field);
                std::string value;
                fs >> value;
                vals2.emplace_back(value);
            }

            if (vals1.size() != vals2.size())
                return false;

            for (size_t i = 0; i < vals1.size(); i++) {
                try {
                    float v1 = std::stof(vals1[i]);
                    float v2 = std::stof(vals2[i]);
                    if (abs(v2 - v1) > 0.00001f)
                        return false;
                } catch (...) {
                    if (vals1[i] != vals2[i])
                        return false;
                }
            }
            return true;
        };
        std::vector<std::string> err_log;
        CNNNetwork network(newNetwork);
        CNNNetwork refNetwork(oldNetwork);
        if (newNetwork->layerCount() != oldNetwork.layerCount())
            THROW_IE_EXCEPTION << "ICNNNetworks have different numbers of layers! " + std::to_string(newNetwork->layerCount()) + " and " + std::to_string(oldNetwork.layerCount());
        auto newIterator = network.begin();
        auto oldIterator = refNetwork.begin();
        for (; newIterator != network.end() && oldIterator != refNetwork.end(); newIterator++, oldIterator++) {
            CNNLayerPtr layer = *newIterator;
            CNNLayerPtr oldLayer = *oldIterator;
            if (layer->type != oldLayer->type) {
                err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different type: " + layer->type + " and " + oldLayer->type);
            }
            if (layer->precision != oldLayer->precision) {
                err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different precision");
            }
            bool success = layer->type == oldLayer->type &&
                layer->insData.size() == oldLayer->insData.size() &&
                layer->outData.size() == oldLayer->outData.size() &&
                layer->precision == oldLayer->precision;

            for (size_t i = 0; i < layer->insData.size() && success; i++) {
                auto lockedOldData = oldLayer->insData[i].lock();
                auto lockedData = layer->insData[i].lock();
                success = success && lockedOldData->getTensorDesc() == lockedData->getTensorDesc();
                if (lockedOldData->getTensorDesc() != lockedData->getTensorDesc()) {
                    err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different tensor desc for locked data");
                }
            }
            for (size_t i = 0; i < layer->outData.size() && success; i++) {
                if (oldLayer->outData[i]->getTensorDesc() != layer->outData[i]->getTensorDesc()) {
                    err_log.push_back("Layer " + oldLayer->name + " and old layer " + oldLayer->name + " have different tensor desc");
                }
                success = success && oldLayer->outData[i]->getTensorDesc() == layer->outData[i]->getTensorDesc();
            }
            for (const auto& item : layer->params) {
                if (!success)
                    break;
                if (oldLayer->params.find(item.first) != oldLayer->params.end()) {
                    if (!compareParamVal(oldLayer->params[item.first], item.second)) {
                        success = false;
                        err_log.push_back("Layer " + layer->name + " in new network differ from reference parameter " + item.first + " (new, old): " + item.second + ", " + oldLayer->params[item.first]);
                    }
                } else {
                    success = false;
                    err_log.push_back("Layer " + oldLayer->name + " in old net has no " + item.first + " attribute.");
                }
            }

            if (!success) {
                for (auto & it: err_log) {
                    std::cout << "ERROR: " << it << std::endl;
                }
                THROW_IE_EXCEPTION << "ICNNNetworks have different layers!";
            }
        }

        InputsDataMap newInput;
        OutputsDataMap newOutput;
        newNetwork->getInputsInfo(newInput);
        newNetwork->getOutputsInfo(newOutput);
        InputsDataMap oldInput = oldNetwork.getInputsInfo();
        OutputsDataMap oldOutput = oldNetwork.getOutputsInfo();

        bool success = newInput.size() == oldInput.size();
        for (const auto& it : newInput) {
            if (!success)
                break;
            success = success && oldInput.find(it.first) != oldInput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different inputs!";

        success = newOutput.size() == oldOutput.size();
        for (const auto& it : newOutput) {
            if (!success)
                break;
            success = success && oldOutput.find(it.first) != oldOutput.end();
        }
        if (!success)
            THROW_IE_EXCEPTION << "ICNNNetworks have different outputs!";
    }
};