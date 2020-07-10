// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers.h"

#include <map>
#include <memory>
#include <string>

#include "blob_factory.hpp"
#include "cnn_network_ngraph_impl.hpp"

using namespace InferenceEngine;

Blob::Ptr Blob::CreateFromData(const DataPtr& data) {
    return CreateBlobFromData(data);
}

struct Data::Impl {
    /**
     * @brief A pointer to the layer that creates this data element, null for input data elements
     */
    CNNLayerWeakPtr creatorLayer;
    
    /**
     * @brief A map of layers that use this node as input.
     * It is useful for recursive NN graph traversal.
     */
    std::map<std::string, CNNLayerPtr> inputTo;
};

Data::Data(const std::string& name, Precision _precision, Layout layout)
    : name(name), userObject({0}), tensorDesc(_precision, layout) {
    _impl = std::make_shared<Impl>();
}

Data::Data(const std::string& name, const TensorDesc& desc): name(name), userObject({0}), tensorDesc(desc) {
    _impl = std::make_shared<Impl>();
}

const Precision& Data::getPrecision() const {
    return tensorDesc.getPrecision();
}

const TensorDesc& Data::getTensorDesc() const {
    return tensorDesc;
}

bool Data::isInitialized() const {
    return !tensorDesc.getDims().empty() || tensorDesc.getLayout() == SCALAR;
}

void Data::setDims(const SizeVector& a_dims) {
    tensorDesc.setDims(a_dims);
}

void Data::setLayout(Layout layout) {
    tensorDesc.setLayout(layout);
}

void Data::reshape(const SizeVector& a_dims, Layout a_layout) {
    tensorDesc.reshape(a_dims, a_layout);
}

Data::Data(const Data& data) :
    name(data.name), userObject(data.userObject), tensorDesc(data.tensorDesc) {
    _impl = std::make_shared<Impl>();
    _impl->creatorLayer = data._impl->creatorLayer;
    _impl->inputTo = data._impl->inputTo;
}

Data & Data::operator = (const Data& data) {
    if (this != &data) {
        name = data.name;
        userObject = data.userObject;
        tensorDesc = data.tensorDesc;

        _impl->creatorLayer = data._impl->creatorLayer;
        _impl->inputTo = data._impl->inputTo;
    }

    return *this;
}

const std::string& Data::getName() const {
    return name;
}

void Data::setName(const std::string& newName) {
    name = newName;
}

const UserValue& Data::getUserObject() const {
    return userObject;
}

Layout Data::getLayout() const {
    return tensorDesc.getLayout();
}

void Data::setPrecision(const Precision& precision) {
    tensorDesc.setPrecision(precision);
}

const SizeVector& Data::getDims() const {
    return tensorDesc.getDims();
}

// compatibility

CNNLayerWeakPtr& InferenceEngine::getCreatorLayer(const DataPtr & data) {
    if (auto ndata = std::dynamic_pointer_cast<details::NGraphData>(data)) {
        return ndata->getCreatorLayer();
    } else {
        return data->_impl->creatorLayer;
    }
}

std::map<std::string, CNNLayerPtr>& InferenceEngine::getInputTo(const DataPtr & data) {
    return data->_impl->inputTo;
}

std::map<std::string, CNNLayerPtr>& InferenceEngine::getInputTo(Data * data) {
    return data->_impl->inputTo;
}

CNNLayerWeakPtr& details::NGraphData::getCreatorLayer() {
    return _impl->creatorLayer;
}

std::map<std::string, CNNLayerPtr>& details::NGraphData::getInputTo() {
    return _impl->inputTo;
}

void details::NGraphData::convertToCNNNetworkImpl() {
    if (network != nullptr) {
        network->convertToCNNNetworkImpl();
    }
}
