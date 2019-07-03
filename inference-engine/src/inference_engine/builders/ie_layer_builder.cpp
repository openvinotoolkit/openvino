// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_layer_builder.hpp>
#include <details/caseless.hpp>

#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <map>

using namespace InferenceEngine;

Builder::Layer::Layer(const std::string& type, const std::string& name):
        name(name), type(type), id((std::numeric_limits<idx_t>::max)()) {}

Builder::Layer::Layer(const ILayer::CPtr& layer) {
    id = layer->getId();
    name = layer->getName();
    type = layer->getType();
    inPorts = layer->getInputPorts();
    outPorts = layer->getOutputPorts();
    params = layer->getParameters();
}

Builder::Layer::Layer(idx_t id, const Builder::Layer& layer): Layer(layer) {
    this->id = id;
}

idx_t Builder::Layer::getId() const noexcept {
    return id;
}

const std::string& Builder::Layer::getType() const noexcept {
    return type;
}
Builder::Layer& Builder::Layer::setType(const std::string& type) {
    this->type = type;
    return *this;
}

const std::string& Builder::Layer::getName() const noexcept {
    return name;
}
Builder::Layer& Builder::Layer::setName(const std::string& name) {
    this->name = name;
    return *this;
}

const std::map<std::string, Parameter>& Builder::Layer::getParameters() const noexcept {
    return params;
}
std::map<std::string, Parameter>& Builder::Layer::getParameters() {
    return params;
}
Builder::Layer& Builder::Layer::setParameters(const std::map<std::string, Parameter>& params) {
    getParameters() = params;
    return *this;
}

std::vector<Port>& Builder::Layer::getInputPorts() {
    return inPorts;
}
const std::vector<Port>& Builder::Layer::getInputPorts() const noexcept {
    return inPorts;
}
Builder::Layer& Builder::Layer::setInputPorts(const std::vector<Port> &ports) {
    getInputPorts() = ports;
    return *this;
}

std::vector<Port>& Builder::Layer::getOutputPorts() {
    return outPorts;
}
const std::vector<Port>& Builder::Layer::getOutputPorts() const noexcept {
    return outPorts;
}
Builder::Layer& Builder::Layer::setOutputPorts(const std::vector<Port> &ports) {
    getOutputPorts() = ports;
    return *this;
}

const ILayer::CPtr Builder::Layer::build() const {
    validate(true);
    return std::static_pointer_cast<const ILayer>(shared_from_this());
}

void Builder::Layer::addValidator(const std::string &type, const std::function<void(const Layer::CPtr&, bool)>& validator) {
    auto holder = getValidatorsHolder();
    if (holder->validators.find(type) == holder->validators.end())
        holder->validators[type] = validator;
}

void Builder::Layer::validate(bool partial) const {
    if (getValidatorsHolder()->validators.find(type) != getValidatorsHolder()->validators.end())
        getValidatorsHolder()->validators[type](shared_from_this(), partial);
}

std::shared_ptr<Builder::ValidatorsHolder> Builder::Layer::getValidatorsHolder() {
    static std::shared_ptr<ValidatorsHolder> localHolder;
    if (localHolder == nullptr) {
        localHolder = std::make_shared<ValidatorsHolder>();
    }
    return localHolder;
}
