// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_cnn_layer_builder.h>

using namespace InferenceEngine;

std::map<std::string, std::string> Builder::convertParameters2Strings(const std::map<std::string, Parameter>& parameters) {
    std::map<std::string, std::string> oldParams;
    for (const auto& param : parameters) {
        // skip blobs and ports
        if (param.second.is<Blob::CPtr>() || param.second.is<Blob::Ptr>() || param.second.is<std::vector<Port>>()
                || param.second.is<PreProcessInfo>())
            continue;
        if (param.second.is<std::string>() || param.second.is<std::vector<std::string>>()) {
            oldParams[param.first] = Builder::convertParameter2String<std::string>(param.second);
        } else if (param.second.is<int>() || param.second.is<std::vector<int>>()) {
            oldParams[param.first] = Builder::convertParameter2String<int>(param.second);
        } else if (param.second.is<float>() || param.second.is<std::vector<float>>()) {
            oldParams[param.first] = Builder::convertParameter2String<float>(param.second);
        } else if (param.second.is<unsigned int>() || param.second.is<std::vector<unsigned int>>()) {
            oldParams[param.first] = Builder::convertParameter2String<unsigned int>(param.second);
        } else if (param.second.is<size_t>() || param.second.is<std::vector<size_t>>()) {
            oldParams[param.first] = Builder::convertParameter2String<size_t>(param.second);
        } else if (param.second.is<bool>() || param.second.is<std::vector<bool>>()) {
            oldParams[param.first] = Builder::convertParameter2String<bool>(param.second);
        } else {
            THROW_IE_EXCEPTION << "Parameter " << param.first << " has unsupported parameter type!";
        }
    }
    return oldParams;
}

Builder::Layer Builder::builderFromCNNLayer(const CNNLayerPtr& cnnLayer) {
    Builder::Layer layer(cnnLayer->type, cnnLayer->name);
    std::vector<Port> inputPorts;
    for (const auto& data : cnnLayer->insData) {
        auto lockedData = data.lock();
        if (!lockedData)
            continue;
        inputPorts.emplace_back(lockedData->getTensorDesc().getDims());
    }

    std::vector<Port> outputPorts;
    for (const auto& data : cnnLayer->outData) {
        outputPorts.emplace_back(data->getTensorDesc().getDims());
    }

    size_t inputsCount = inputPorts.size();
    std::map<std::string, Blob::Ptr> blobs = cnnLayer->blobs;
    if (blobs.find("weights") != blobs.end()) {
        auto port = Port();
        port.setParameter("type", "weights");
        inputPorts.push_back(port);
    }
    if (blobs.find("biases") != blobs.end()) {
        if (inputsCount == inputPorts.size()) {
            auto port = Port();
            port.setParameter("type", "weights");
            inputPorts.push_back(port);
        }

        auto port = Port();
        port.setParameter("type", "biases");
        inputPorts.push_back(port);
    }
    for (const auto& it : blobs) {
        if (it.first == "weights" || it.first == "biases")
            continue;
        auto port = Port();
        port.setParameter("type", it.first);
        inputPorts.emplace_back(port);
    }

    std::map<std::string, Parameter> params;
    for (const auto& it : cnnLayer->params) {
        params[it.first] = it.second;
    }

    layer.setInputPorts(inputPorts).setOutputPorts(outputPorts).setParameters(params);

    Builder::ConverterRegister::convert(cnnLayer, layer);

    return layer;
}

Builder::ConverterRegister::ConverterRegister(const std::string& type, const std::function<void(const CNNLayerPtr&, Layer&)>& converter) {
    if (getConvertersHolder().converters.find(type) == getConvertersHolder().converters.end())
        getConvertersHolder().converters[type] = converter;
}

Builder::ConvertersHolder &Builder::ConverterRegister::getConvertersHolder() {
    static Builder::ConvertersHolder holder;
    return holder;
}
