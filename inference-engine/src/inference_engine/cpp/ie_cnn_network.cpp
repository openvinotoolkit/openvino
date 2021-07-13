// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp/ie_cnn_network.h"
#include "exception2status.hpp"

#include "cnn_network_ngraph_impl.hpp"
#include "ie_itt.hpp"

namespace InferenceEngine {

CNNNetwork::CNNNetwork() :
    network(), actual() {
}

IE_SUPPRESS_DEPRECATED_START

CNNNetwork::CNNNetwork(std::shared_ptr<ICNNNetwork> network)
    : network(network) {
    actual = network.get();
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
}

CNNNetwork::CNNNetwork(const std::shared_ptr<ngraph::Function>& graph,
                       const std::vector<IExtensionPtr>& exts) {
    OV_ITT_SCOPED_TASK(itt::domains::IE, "CNNNetwork::CNNNetwork");

    if (graph == nullptr) {
        IE_THROW() << "CNNNetwork was not initialized: 'graph' object is empty";
    }

    // Create CNNNetworkNGraphImpl
    network = std::make_shared<details::CNNNetworkNGraphImpl>(graph, exts);
    actual = network.get();
    if (actual == nullptr) {
        IE_THROW() << "CNNNetwork was not initialized.";
    }
}

OutputsDataMap CNNNetwork::getOutputsInfo() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    OutputsDataMap outputs;
    actual->getOutputsInfo(outputs);
    return outputs;
}

InputsDataMap CNNNetwork::getInputsInfo() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    InputsDataMap inputs;
    actual->getInputsInfo(inputs);
    return inputs;
}

size_t CNNNetwork::layerCount() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return actual->layerCount();
}

const std::string& CNNNetwork::getName() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return actual->getName();
}

void CNNNetwork::setBatchSize(const size_t size) {
    CALL_STATUS_FNC(setBatchSize, size);
}

size_t CNNNetwork::getBatchSize() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return actual->getBatchSize();
}

CNNNetwork::operator ICNNNetwork::Ptr() {
    return network;
}

CNNNetwork::operator ICNNNetwork&() {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return *actual;
}

CNNNetwork::operator const ICNNNetwork&() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return *actual;
}

std::shared_ptr<ngraph::Function> CNNNetwork::getFunction() {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return actual->getFunction();
}

std::shared_ptr<const ngraph::Function> CNNNetwork::getFunction() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    return actual->getFunction();
}

void CNNNetwork::addOutput(const std::string& layerName, size_t outputIndex) {
    CALL_STATUS_FNC(addOutput, layerName, outputIndex);
}

ICNNNetwork::InputShapes CNNNetwork::getInputShapes() const {
    if (actual == nullptr) IE_THROW() << "CNNNetwork was not initialized.";
    ICNNNetwork::InputShapes shapes;
    InputsDataMap inputs;
    actual->getInputsInfo(inputs);
    for (const auto& pair : inputs) {
        auto info = pair.second;
        if (info) {
            auto data = info->getInputData();
            if (data) {
                shapes[data->getName()] = data->getTensorDesc().getDims();
            }
        }
    }
    return shapes;
}

void CNNNetwork::reshape(const ICNNNetwork::InputShapes& inputShapes) {
    CALL_STATUS_FNC(reshape, inputShapes);
}

void CNNNetwork::serialize(const std::string& xmlPath, const std::string& binPath) const {
    CALL_STATUS_FNC(serialize, xmlPath, binPath);
}

void CNNNetwork::serialize(std::ostream& xmlBuf, std::ostream& binBuf) const {
    CALL_STATUS_FNC(serialize, xmlBuf, binBuf);
}

void CNNNetwork::serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob) const {
    CALL_STATUS_FNC(serialize, xmlBuf, binBlob);
}

std::string CNNNetwork::getOVNameForTensor(const std::string& orig_name) const {
    std::string ov_name;
    CALL_STATUS_FNC(getOVNameForTensor, ov_name, orig_name);
    return ov_name;
}

}  // namespace InferenceEngine
