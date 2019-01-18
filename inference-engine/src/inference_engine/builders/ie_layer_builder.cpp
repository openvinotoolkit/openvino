// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_layer_builder.hpp>
#include <details/caseless.hpp>
#include <ie_network.hpp>

#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <map>

using namespace InferenceEngine;

Builder::Layer::Layer(const std::string& type, const std::string& name): id((std::numeric_limits<idx_t>::max)()), type(type), name(name) {}

Builder::Layer::Layer(const ILayer::Ptr& layer) {
    id = layer->getId();
    getType() = layer->getType();
    getName() = layer->getName();
    getGraph() = layer->getGraph();
    getParameters() = layer->getParameters()->getParameters();
    getInputPorts() = layer->getInputPorts();
    getOutputPorts() = layer->getOutputPorts();
    getConstantData() = layer->getParameters()->getConstantData();
}
Builder::Layer::Layer(const ILayer::CPtr& layer) {
    id = layer->getId();
    getType() = layer->getType();
    getName() = layer->getName();
    getGraph() = layer->getGraph();
    getParameters() = layer->getParameters()->getParameters();
    getInputPorts() = layer->getInputPorts();
    getOutputPorts() = layer->getOutputPorts();
    getConstantData() = layer->getParameters()->getConstantData();
}

Builder::Layer::Layer(idx_t id, const Builder::Layer& layer): Layer(layer) {
    this->id = id;
}

idx_t Builder::Layer::getId() const {
    return id;
}

std::string& Builder::Layer::getType() {
    return type;
}
const std::string& Builder::Layer::getType() const {
    return type;
}
Builder::Layer& Builder::Layer::setType(const std::string& type) {
    getType() = type;
    return *this;
}

std::string& Builder::Layer::getName() {
    return name;
}
const std::string& Builder::Layer::getName() const {
    return name;
}
Builder::Layer& Builder::Layer::setName(const std::string& name) {
    getName() = name;
    return *this;
}

INetwork::Ptr& Builder::Layer::getGraph() {
    return graph;
}
const INetwork::Ptr& Builder::Layer::getGraph() const {
    return graph;
}
Builder::Layer& Builder::Layer::setGraph(const INetwork::Ptr& graph) {
    getGraph() = graph;
    return *this;
}

const std::map<std::string, Parameter>& Builder::Layer::getParameters() const {
    return params;
}
std::map<std::string, Parameter>& Builder::Layer::getParameters() {
    return params;
}
Builder::Layer& Builder::Layer::setParameters(const std::map<std::string, Parameter>& params) {
    getParameters() = params;
    return *this;
}

const std::map<std::string, Blob::CPtr>& Builder::Layer::getConstantData() const {
    return constData;
}
std::map<std::string, Blob::CPtr>& Builder::Layer::getConstantData() {
    return constData;
}
Builder::Layer& Builder::Layer::setConstantData(const std::map<std::string, Blob::Ptr>& constData) {
    for (const auto& it : constData)
        addConstantData(it.first, it.second);
    return *this;
}
Builder::Layer& Builder::Layer::setConstantData(const std::map<std::string, Blob::CPtr>& constData) {
    getConstantData() = constData;
    return *this;
}
Builder::Layer& Builder::Layer::addConstantData(const std::string& name, const Blob::CPtr& data) {
    getConstantData()[name] = data;
    return *this;
}

std::vector<Port>& Builder::Layer::getInputPorts() {
    return inPorts;
}
const std::vector<Port>& Builder::Layer::getInputPorts() const {
    return inPorts;
}
Builder::Layer& Builder::Layer::setInputPorts(const std::vector<Port> &ports) {
    getInputPorts() = ports;
    return *this;
}

std::vector<Port>& Builder::Layer::getOutputPorts() {
    return outPorts;
}
const std::vector<Port>& Builder::Layer::getOutputPorts() const {
    return outPorts;
}
Builder::Layer& Builder::Layer::setOutputPorts(const std::vector<Port> &ports) {
    getOutputPorts() = ports;
    return *this;
}

const ILayer::Ptr Builder::Layer::build() const {
    validate();
    details::Layer::Ptr layer = std::make_shared<details::Layer>(id);

    layer->getName() = name;
    layer->getType() = type;
    layer->setGraph(graph);
    layer->getInputPorts() = inPorts;
    layer->getOutputPorts() = outPorts;
    layer->getParameters()->getParameters() = params;
    layer->getParameters()->getConstantData() = constData;
    return std::static_pointer_cast<ILayer>(layer);
}

void Builder::Layer::addValidator(const std::string &type, const std::function<void(const Layer&)>& validator) {
    auto holder = getValidatorsHolder();
    if (holder->validators.find(type) == holder->validators.end())
        holder->validators[type] = validator;
}

void Builder::Layer::validate() const {
    if (getValidatorsHolder()->validators.find(type) != getValidatorsHolder()->validators.end())
        getValidatorsHolder()->validators[type](*this);
}

std::shared_ptr<Builder::ValidatorsHolder> Builder::Layer::getValidatorsHolder() {
    static std::shared_ptr<ValidatorsHolder> localHolder;
    if (localHolder == nullptr) {
        localHolder = std::make_shared<ValidatorsHolder>();
    }
    return localHolder;
}
