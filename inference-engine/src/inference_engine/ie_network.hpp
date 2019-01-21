// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_inetwork.hpp>
#include <ie_blob.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace InferenceEngine {
namespace details {

class Network;

class Parameters: public IParameters {
public:
    using Ptr = std::shared_ptr<Parameters>;

    const std::map<std::string, Parameter>& getParameters() const noexcept override {
        return params;
    }
    const std::map<std::string, Blob::CPtr>& getConstantData() const noexcept override {
        return constData;
    }

    std::map<std::string, Parameter>& getParameters() {
        return params;
    }
    std::map<std::string, Blob::CPtr>& getConstantData() noexcept {
        return constData;
    }
private:
    std::map<std::string, Parameter> params;
    std::map<std::string, InferenceEngine::Blob::CPtr> constData;
};

class Layer: public ILayer {
public:
    using Ptr = std::shared_ptr<Layer>;

    explicit Layer(size_t id): id(id), params(new Parameters()) {}
    Layer(const Layer& layer) {
        this->outputs = layer.getOutputPorts();
        this->inputs = layer.getInputPorts();
        this->params = layer.getParameters();
        this->subGraph = layer.getGraph();
        this->name = layer.getName();
        this->type = layer.getType();
        this->id = layer.getId();
    }
    explicit Layer(const ILayer& layer) {
        this->outputs = layer.getOutputPorts();
        this->inputs = layer.getInputPorts();
        this->params = layer.getParameters();
        this->subGraph = layer.getGraph();
        this->name = layer.getName();
        this->type = layer.getType();
        this->id = layer.getId();
    }

    size_t getId() const noexcept override {
        return id;
    }
    const std::string& getName() const noexcept override {
        return name;
    }
    const std::string& getType() const noexcept override {
        return type;
    }
    const INetwork::Ptr& getGraph() const noexcept override {
        return subGraph;
    }
    const IParameters::Ptr& getParameters() const noexcept override {
        return params;
    }
    const std::vector<Port>& getInputPorts() const noexcept override {
        return inputs;
    }
    const std::vector<Port>& getOutputPorts() const noexcept override {
        return outputs;
    }

    std::string& getName() noexcept {
        return name;
    }

    std::string& getType() noexcept {
        return type;
    }
    std::shared_ptr<Network> getGraph() noexcept {
        return std::dynamic_pointer_cast<Network>(subGraph);
    }
    void setGraph(const INetwork::Ptr& graph) noexcept {
        subGraph = graph;
    }
    Parameters::Ptr getParameters() noexcept {
        return std::dynamic_pointer_cast<Parameters>(params);
    }
    std::vector<Port>& getInputPorts() noexcept {
        return inputs;
    }
    std::vector<Port>& getOutputPorts() noexcept {
        return outputs;
    }

private:
    idx_t id;
    std::string name;
    std::string type;
    INetwork::Ptr subGraph;
    IParameters::Ptr params;
    std::vector<Port> inputs;
    std::vector<Port> outputs;
};

class Network: public INetwork {
public:
    using Ptr = std::shared_ptr<Network>;
    using iterator = details::INetworkIterator<Network, Layer>;

    explicit Network(const Context& context, const std::string& name = "");
    Network(const Context& context, const INetwork& network);
    Network(const Context& context, const Network& network);

    Network& operator=(const Network& network);
    Network& operator=(const INetwork& network);

    const_iterator begin() const noexcept override;
    const_iterator end() const noexcept override;
    iterator begin() noexcept;
    iterator end() noexcept;

    const ILayer::Ptr getLayer(size_t id) const noexcept override;
    const std::vector<ILayer::Ptr> getInputs() const noexcept override;
    const std::vector<ILayer::Ptr> getOutputs() const noexcept override;
    const std::vector<Connection> getLayerConnections(idx_t layerId) const noexcept override;
    size_t size() const noexcept override;
    const std::string& getName() const noexcept override;
    const Context& getContext() const noexcept override;

    const std::vector<Connection>& getConnections() const noexcept;
    Layer::Ptr getLayer(size_t id) noexcept;
    std::string& getName() noexcept;

    void addLayer(const ILayer::Ptr& layer) noexcept;
    void addConnection(const Connection& connection) noexcept;

private:
    const Context ctx;
    std::string name;
    std::vector<Layer::Ptr> layers;
    std::vector<Connection> connections;
};

}  // namespace details
}  // namespace InferenceEngine
