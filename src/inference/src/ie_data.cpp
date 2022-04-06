// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <string>

#include "blob_factory.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "ie_api.h"
#include "ie_common.h"

using namespace InferenceEngine;

Blob::Ptr Blob::CreateFromData(const DataPtr& data) {
    // TODO Here some decision should be made about the layout.
    // For now we just pass the layout and use conversion to NCHW for ANY.
    InferenceEngine::Layout targetLayout = data->getLayout();
    if (data->getLayout() == InferenceEngine::Layout::ANY) {
        targetLayout = InferenceEngine::Layout::NCHW;
    }

    InferenceEngine::TensorDesc desc(data->getPrecision(), data->getTensorDesc().getDims(), targetLayout);

    switch (data->getPrecision()) {
    case InferenceEngine::Precision::FP32:
        return std::make_shared<InferenceEngine::TBlob<float>>(desc);
    case InferenceEngine::Precision::Q78:
    case InferenceEngine::Precision::I16:
    case InferenceEngine::Precision::FP16:
        return std::make_shared<InferenceEngine::TBlob<short>>(desc);
    case InferenceEngine::Precision::U8:
        return std::make_shared<InferenceEngine::TBlob<uint8_t>>(desc);
    case InferenceEngine::Precision::I8:
        return std::make_shared<InferenceEngine::TBlob<int8_t>>(desc);
    case InferenceEngine::Precision::I32:
        return std::make_shared<InferenceEngine::TBlob<int32_t>>(desc);
    case InferenceEngine::Precision::BF16:
        return std::make_shared<InferenceEngine::TBlob<short>>(desc);
    default:
        IE_THROW() << "precision is no set";
    }
}

namespace InferenceEngine {

class CNNLayer;

/**
 * @brief A smart pointer to the CNNLayer
 */
using CNNLayerPtr = std::shared_ptr<CNNLayer>;
/**
 * @brief A smart weak pointer to the CNNLayer
 */
using CNNLayerWeakPtr = std::weak_ptr<CNNLayer>;

}  // namespace InferenceEngine

class Data::Impl {
public:
    /**
     * @brief A pointer to the layer that creates this data element, null for input data elements
     */
    CNNLayerWeakPtr creatorLayer;

    /**
     * @brief A map of layers that use this node as input.
     * It is useful for recursive NN graph traversal.
     */
    std::map<std::string, CNNLayerPtr> inputTo;

    ngraph::PartialShape pShape;
};

Data::Data(const std::string& name, Precision _precision, Layout layout)
    : name(name),
      userObject({0}),
      tensorDesc(_precision, layout) {
    _impl = std::make_shared<Impl>();
}

Data::Data(const std::string& name, const TensorDesc& desc) : name(name), userObject({0}), tensorDesc(desc) {
    _impl = std::make_shared<Impl>();
    _impl->pShape = ngraph::PartialShape(desc.getDims());
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
    _impl->pShape = ngraph::PartialShape(a_dims);
}

void Data::setLayout(Layout layout) {
    tensorDesc.setLayout(layout);
}

void Data::reshape(const SizeVector& a_dims, Layout a_layout) {
    tensorDesc.reshape(a_dims, a_layout);
    _impl->pShape = ngraph::PartialShape(a_dims);
}

Data::Data(const Data& data) : name(data.name), userObject(data.userObject), tensorDesc(data.tensorDesc) {
    _impl = std::make_shared<Impl>();
    _impl->creatorLayer = data._impl->creatorLayer;
    _impl->inputTo = data._impl->inputTo;
    _impl->pShape = data._impl->pShape;
}

Data& Data::operator=(const Data& data) {
    if (this != &data) {
        name = data.name;
        userObject = data.userObject;
        tensorDesc = data.tensorDesc;

        _impl->creatorLayer = data._impl->creatorLayer;
        _impl->inputTo = data._impl->inputTo;
        _impl->pShape = data._impl->pShape;
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
    if (_impl->pShape.is_dynamic())
        IE_THROW() << "Cannot return dims for Data with dynamic shapes!";
    if (tensorDesc.getDims().empty() && tensorDesc.getLayout() != SCALAR) {
        tensorDesc.setDims(_impl->pShape.to_shape());
    }
    return tensorDesc.getDims();
}

// compatibility

namespace InferenceEngine {

INFERENCE_ENGINE_API_CPP(CNNLayerWeakPtr&) getCreatorLayer(const DataPtr& data);
INFERENCE_ENGINE_API_CPP(std::map<std::string, CNNLayerPtr>&) getInputTo(const DataPtr& data);
INFERENCE_ENGINE_API_CPP(std::map<std::string, CNNLayerPtr>&) getInputTo(Data* data);

CNNLayerWeakPtr& getCreatorLayer(const DataPtr& data) {
    return data->_impl->creatorLayer;
}

std::map<std::string, CNNLayerPtr>& getInputTo(const DataPtr& data) {
    return data->_impl->inputTo;
}

std::map<std::string, CNNLayerPtr>& getInputTo(Data* data) {
    return data->_impl->inputTo;
}

}  // namespace InferenceEngine
