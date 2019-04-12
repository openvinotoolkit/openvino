// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transform/transform_network.hpp>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <map>

using namespace InferenceEngine;

Transform::Port::Port(Builder::Network& network, PortInfo port, bool isInput)
    : network(network), port(port), input(isInput) {
    const auto& layer = network.getLayer(port.layerId());
    if (isInput) {
        if (layer->getInputPorts().size() < port.portId())
            THROW_IE_EXCEPTION << "Cannot find input port "
                               << port.portId() << " in layer "
                               << layer->getName();
    } else {
        if (layer->getOutputPorts().size() < port.portId())
            THROW_IE_EXCEPTION << "Cannot find output port "
                               << port.portId() << " in layer "
                               << layer->getName();
    }
}

PortData::Ptr Transform::Port::getData() const {
    return input ?
           network.getLayer(port.layerId())->getInputPorts()[port.portId()].getData() :
           network.getLayer(port.layerId())->getOutputPorts()[port.portId()].getData();
}

const std::map<std::string, Parameter> &Transform::Port::getParameters() const {
    return input ?
           network.getLayer(port.layerId())->getInputPorts()[port.portId()].getParameters() :
           network.getLayer(port.layerId())->getOutputPorts()[port.portId()].getParameters();
}

Transform::Layer Transform::Port::getLayer() const {
    return Transform::Network(network).getLayer(getPortInfo().layerId());
}

Transform::Connection Transform::Port::getConnection() const {
    return Connection(*this);
}

void Transform::Port::connect(const Port& port) {
    if (this->input)
        this->getConnection().setSource(port);
    else
        this->getConnection().addDestination(port);
}

void Transform::Port::disconnect() {
    getConnection().remove();
}

const SizeVector& Transform::Port::shape() const {
    return this->getData()->getData()->getTensorDesc().getDims();
}

PortInfo Transform::Port::getPortInfo() const {
    return port;
}

bool Transform::Port::operator==(const Port& rObj) const {
    return &network == &rObj.network &&
           port == rObj.port &&
           input == rObj.input;
}

bool Transform::Port::operator!=(const Port& rObj) const {
    return !(*this == rObj);
}


Transform::Layer::Layer(Builder::Network& network, idx_t id)
    : network(network), layerId(id) {}

idx_t Transform::Layer::getId() const {
    return layerId;
}

std::string Transform::Layer::getName() const {
    return getLayer()->getName();
}

std::string Transform::Layer::getType() const {
    return getLayer()->getType();
}

Builder::Layer::Ptr Transform::Layer::getLayer() const {
    return network.getLayer(layerId);
}

Transform::Layer::operator Builder::Layer::Ptr() const {
    return getLayer();
}

Transform::Port Transform::Layer::getInPort() const {
    if (getLayer()->getInputPorts().size() != 1)
        THROW_IE_EXCEPTION << "Layer " << getName()
                           << " has more than 1 input port.";
    return Transform::Port(network, {layerId, 0}, true);
}

Transform::Port Transform::Layer::getInPort(idx_t idx) const {
    if (getLayer()->getInputPorts().size() <= idx)
        THROW_IE_EXCEPTION << "Layer " << getName()
                           << " has less than " << idx << " input port(s).";
    return Transform::Port(network, {layerId, idx}, true);
}

std::vector<Transform::Port> Transform::Layer::getInPorts() const {
    std::vector<Transform::Port> ports;
    for (size_t i = 0; i < getLayer()->getInputPorts().size(); i++) {
        ports.push_back({network, {layerId, i}, true});
    }
    return ports;
}

Transform::Port Transform::Layer::getOutPort() const {
    if (getLayer()->getOutputPorts().size() != 1)
        THROW_IE_EXCEPTION << "Layer " << getName()
                           << " has more than 1 output port.";
    return Transform::Port(network, {layerId, 0}, false);
}

Transform::Port Transform::Layer::getOutPort(idx_t idx) const {
    if (getLayer()->getOutputPorts().size() <= idx)
        THROW_IE_EXCEPTION << "Layer " << getName()
                           << " has less than " << idx << " output port(s).";
    return Transform::Port(network, {layerId, idx}, false);
}

std::vector<Transform::Port> Transform::Layer::getOutPorts() const {
    std::vector<Transform::Port> ports;
    for (size_t i = 0; i < getLayer()->getInputPorts().size(); i++) {
        ports.push_back({network, {layerId, i}, false});
    }
    return ports;
}

void Transform::Layer::setParameter(const std::string& key, const Parameter& value) {
    auto& params = getLayer()->getParameters();
    params[key] = value;
}

Parameter& Transform::Layer::getParameter(const std::string& key) const {
    auto& params = getLayer()->getParameters();
    if (params.find(key) == params.end())
        THROW_IE_EXCEPTION << "Layer " << getName() << " has no parameter " << key;
    return params[key];
}

Transform::Connection::Connection(const Transform::Port& port)
    : network(port.network), inPort({(std::numeric_limits<idx_t>::max)(), (std::numeric_limits<idx_t>::max)()}) {
    if (port.input) {
        outPorts = {port.getPortInfo()};
        for (const auto& connection : network.getLayerConnections(port.getPortInfo().layerId())) {
            if (connection.to() == port.getPortInfo()) {
                inPort = connection.from();
                break;
            }
        }
    } else {
        inPort = port.getPortInfo();
        for (const auto& connection : network.getLayerConnections(port.getPortInfo().layerId())) {
            if (connection.from() == port.getPortInfo()) {
                outPorts.emplace_back(connection.to());
            }
        }
    }
}
Transform::Connection::Connection(Builder::Network& network, const InferenceEngine::Connection& connection)
    : Connection(network, connection.from(), connection.to()) {}
Transform::Connection::Connection(Builder::Network& network, const PortInfo& inPort, const PortInfo& outPort)
    : Connection(network, inPort, std::vector<PortInfo>({outPort})) {}
Transform::Connection::Connection(Builder::Network& network, const PortInfo& inPort, const std::vector<PortInfo>& outPorts)
    : network(network), inPort(inPort), outPorts(outPorts) {}

Transform::Port Transform::Connection::getSource() const {
    if (!inPortExist())
        THROW_IE_EXCEPTION << "Connection doesn't have source port!";
    return Port(network, inPort, false);
}

void Transform::Connection::setSource(const Transform::Port &port) {
    if (inPortExist()) {
        // disconnect old port
        for (const auto& outPort : outPorts) {
            network.disconnect({inPort, outPort});
        }
    }
    inPort = port.getPortInfo();
    for (const auto& outPort : outPorts) {
        network.connect(inPort, outPort);
    }
}

Transform::Port Transform::Connection::getDestination() const {
    if (outPorts.size() != 1)
        THROW_IE_EXCEPTION << "Connection has more than 1 output.";
    return Transform::Port(network, outPorts[0], true);
}

Transform::Port Transform::Connection::getDestination(idx_t idx) {
    if (outPorts.size() <= idx)
        THROW_IE_EXCEPTION << "Connection has less than "
                           << idx << " input port(s).";
    return Transform::Port(network, outPorts[idx], true);
}

std::vector<Transform::Port> Transform::Connection::getDestinations() const {
    std::vector<Transform::Port> ports;
    for (const auto& port : outPorts) {
        ports.emplace_back(network, port, true);
    }
    return ports;
}

void Transform::Connection::addDestination(const Transform::Port &port) {
    for (const auto& outPort : outPorts) {
        if (outPort == port.getPortInfo()) {
            THROW_IE_EXCEPTION << "Cannot connect twice with one port!";
        }
    }
    outPorts.emplace_back(port.getPortInfo());
    if (!inPortExist())
        return;
    network.connect(inPort, outPorts[outPorts.size() - 1]);
}

void Transform::Connection::setDestination(const Transform::Port &port) {
    if (outPorts.size() > 1) {
        THROW_IE_EXCEPTION << "Cannot set destination for connection which has more than 1 consumer."
                           << "Please use addDestination or setDestinations methods!";
    }

    if (!outPorts.empty()) {
        if (inPortExist())
            network.disconnect({inPort, outPorts[0]});
        outPorts.clear();
    }
    addDestination(port);
}

void Transform::Connection::setDestinations(const std::vector<Transform::Port> &ports) {
    if (!outPorts.empty() && outPorts.size() != ports.size())
        THROW_IE_EXCEPTION << "Cannot change number of output connections!";

    if (inPortExist()) {
        for (const auto &port : outPorts) {
            network.disconnect({inPort, port});
        }
    }
    outPorts.clear();
    for (const auto &port : ports) {
        addDestination(port);
    }
}

void Transform::Connection::remove() {
    if (!inPortExist())
        return;
    for (const auto& port : outPorts) {
        network.disconnect({inPort, port});
    }
}

bool Transform::Connection::inPortExist() const {
    static PortInfo uninitPort((std::numeric_limits<idx_t>::max)(), (std::numeric_limits<idx_t>::max)());
    return inPort != uninitPort;
}

Transform::Layer Transform::Network::addLayer(const Builder::Layer &layer) {
    idx_t layerId = network.addLayer(layer);
    return Transform::Layer(network, layerId);
}

void Transform::Network::removeLayer(const Transform::Layer &layer) {
    for (const auto& connection : network.getLayerConnections(layer.getId()))
        network.disconnect(connection);
    network.removeLayer(layer.getId());
}

Transform::Layer Transform::Network::getLayer(const std::string &name) const {
    for (const auto& layer : network) {
        if (layer->getName() == name)
            return Transform::Layer(network, layer->getId());
    }
    THROW_IE_EXCEPTION << "Layer with name: " << name << " was not found!";
}

Transform::Layer Transform::Network::getLayer(idx_t id) const {
    for (const auto& layer : network) {
        if (layer->getId() == id)
            return Transform::Layer(network, layer->getId());
    }
    THROW_IE_EXCEPTION << "Layer with id: " << id << " was not found!";
}

Transform::Connection Transform::Network::connect(const Transform::Layer &src,
        const Transform::Layer &dst) {
    Port srcPort = src.getOutPort();
    Port dstPort = dst.getInPort();

    network.connect(srcPort.getPortInfo(), dstPort.getPortInfo());
    return Connection(network, srcPort.getPortInfo(), dstPort.getPortInfo());
}

Transform::Connection Transform::Network::connect(const Transform::Port &src,
        const Transform::Port &dst) {
    network.connect(src.getPortInfo(), dst.getPortInfo());
    return Connection(network, src.getPortInfo(), dst.getPortInfo());
}

void Transform::Network::disconnect(const Transform::Layer &src, const Transform::Layer &dst) {
    getConnection(src, dst).remove();
}

void Transform::Network::disconnect(const Transform::Port &src, const Transform::Port &dst) {
    getConnection(src, dst).remove();
}

Builder::Network& Transform::Network::getBuilderNetwork() const {
    return network;
}

Transform::Connection Transform::Network::getConnection(const Transform::Layer &src,
        const Transform::Layer &dst) const {
    Port srcPort = src.getOutPort();
    Port dstPort = dst.getInPort();

    for (const auto& connection : network.getConnections()) {
        if (connection.from() == srcPort.getPortInfo() && connection.to() == dstPort.getPortInfo())
            return Connection(network, srcPort.getPortInfo(), dstPort.getPortInfo());
    }
    THROW_IE_EXCEPTION << "Connection " << src.getName() << " -> " << dst.getName() << " was not found!";
}

Transform::Connection Transform::Network::getConnection(const Transform::Port &src,
        const Transform::Port &dst) const {
    for (const auto& connection : network.getConnections()) {
        if (connection.from() == src.getPortInfo() && connection.to() == dst.getPortInfo())
            return Connection(network, src.getPortInfo(), dst.getPortInfo());
    }
    THROW_IE_EXCEPTION << "Connection " << getLayer(src.getPortInfo().layerId()).getName()
        << " -> " << getLayer(dst.getPortInfo().layerId()).getName() << " was not found!";
}
