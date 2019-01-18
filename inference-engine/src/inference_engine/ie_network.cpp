// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_network.hpp"
#include <details/ie_inetwork_iterator.hpp>
#include <details/caseless.hpp>
#include <iterator>
#include <string>
#include <vector>
#include <memory>

using namespace InferenceEngine;

details::Network &details::Network::operator=(const details::Network &network) {
    if (this == &network)
        return *this;
    name = network.getName();
    for (const auto& layer : network) {
        layers.push_back(Layer::Ptr(new details::Layer(*layer)));
    }
    for (const auto& connection : network.connections) {
        connections.push_back(connection);
    }
    return *this;
}

details::Network &details::Network::operator=(const INetwork &network) {
    if (this == &network)
        return *this;
    name = network.getName();
    for (const auto& layer : network) {
        layers.push_back(std::make_shared<details::Layer>(*layer));
        for (const auto& newConnection : network.getLayerConnections(layer->getId())) {
            bool connectionFound = false;
            for (const auto& connection : connections) {
                if (connection == newConnection) {
                    connectionFound = true;
                    break;
                }
            }
            if (!connectionFound)
                connections.push_back(newConnection);
        }
    }
    return *this;
}

details::Network::Network(const Context& context, const std::string& name): ctx(context), name(name) {}

details::Network::Network(const Context& context, const details::Network &network): ctx(context) {
    *this = network;
}

details::Network::Network(const Context& context, const INetwork &network): ctx(context) {
    *this = network;
}

size_t details::Network::size() const noexcept {
    return static_cast<size_t>(std::distance(std::begin(*this), std::end(*this)));
}

const std::string& details::Network::getName() const noexcept {
    return name;
}

std::string& details::Network::getName() noexcept {
    return name;
}

const Context& details::Network::getContext() const noexcept {
    return ctx;
}

const ILayer::Ptr details::Network::getLayer(size_t id) const noexcept {
    for (const auto& layer : layers) {
        if (layer->getId() == id)
            return std::static_pointer_cast<ILayer>(layer);
    }
    return nullptr;
}

const std::vector<ILayer::Ptr> details::Network::getInputs() const noexcept {
    std::vector<ILayer::Ptr> inputs;
    for (const auto& layer : layers) {
        bool isInputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
            if (connection.to().layerId() == layer->getId()) {
                isInputLayer = false;
                break;
            }
        }
        if (isInputLayer) {
            inputs.push_back(layer);
        }
    }
    return inputs;
}

const std::vector<ILayer::Ptr> details::Network::getOutputs() const noexcept {
    std::vector<ILayer::Ptr> outputs;
    for (const auto& layer : layers) {
        bool isOutputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
            if (connection.from().layerId() == layer->getId()) {
                isOutputLayer = false;
                break;
            }
        }
        if (isOutputLayer) {
            outputs.push_back(layer);
        }
    }
    return outputs;
}

const std::vector<Connection>& details::Network::getConnections() const noexcept {
    return connections;
}

details::Layer::Ptr details::Network::getLayer(size_t id) noexcept {
    for (const auto& layer : layers) {
        if (layer->getId() == id)
            return layer;
    }
    return nullptr;
}

const std::vector<Connection> details::Network::getLayerConnections(idx_t layerId) const noexcept {
    std::vector<Connection> layerConnections;
    for (auto& connection : connections) {
        if (connection.from().layerId() == layerId || connection.to().layerId() == layerId)
            layerConnections.push_back(connection);
    }
    return layerConnections;
}

void details::Network::addLayer(const ILayer::Ptr &layer) noexcept {
    if (layer)
        layers.push_back(std::make_shared<Layer>(*layer));
}

void details::Network::addConnection(const Connection &connection) noexcept {
    connections.push_back(connection);
}

INetwork::const_iterator details::Network::begin() const noexcept {
    return INetwork::const_iterator(this);
}

INetwork::const_iterator details::Network::end() const noexcept {
    return INetwork::const_iterator(this, true);
}

details::Network::iterator details::Network::begin() noexcept {
    return Network::iterator(this);
}

details::Network::iterator details::Network::end() noexcept {
    return Network::iterator(this, true);
}
