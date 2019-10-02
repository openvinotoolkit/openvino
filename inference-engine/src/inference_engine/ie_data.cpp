// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_layers.h"
#include "ie_data.h"
#include "blob_factory.hpp"
#include <memory>
#include <string>
#include <map>

using namespace InferenceEngine;

Blob::Ptr Blob::CreateFromData(const DataPtr &data) {
    return CreateBlobFromData(data);
}

Data::Data(const std::string &name, Precision _precision, Layout layout): name(name), userObject({0}),
                                                                          tensorDesc(_precision, layout) { }

Data::Data(const std::string &name, const SizeVector &a_dims, Precision _precision, Layout layout)
        : name(name), userObject({0}),
          tensorDesc(_precision, SizeVector(a_dims.rbegin(), a_dims.rend()), layout) { }

Data::Data(const std::string &name, const TensorDesc &desc): name(name), userObject({0}),
                                                             tensorDesc(desc) { }

const Precision& Data::getPrecision() const {
    return tensorDesc.getPrecision();
}

const TensorDesc& Data::getTensorDesc() const {
    return tensorDesc;
}

bool Data::isInitialized() const {
    return !tensorDesc.getDims().empty() || tensorDesc.getLayout() == SCALAR;
}

void Data::setDims(const SizeVector &a_dims) {
    tensorDesc.setDims(a_dims);
}

void Data::setLayout(Layout layout) {
    tensorDesc.setLayout(layout);
}

void Data::reshape(const SizeVector &a_dims, Layout a_layout) {
    tensorDesc.reshape(a_dims, a_layout);
}

CNNLayerWeakPtr &Data::getCreatorLayer() {
    return creatorLayer;
}

const std::string &Data::getName() const {
    return name;
}

void Data::setName(const std::string& newName) {
    name = newName;
}

std::map<std::string, CNNLayerPtr> &Data::getInputTo() {
    return inputTo;
}

const UserValue& Data::getUserObject() const {
    return userObject;
}

Layout Data::getLayout() const {
    return tensorDesc.getLayout();
}

void Data::setPrecision(const Precision & precision) {
    tensorDesc.setPrecision(precision);
}

const SizeVector& Data::getDims() const {
    return tensorDesc.getDims();
}
