// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <ie_builders.hpp>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace InferenceEngine {
namespace Transform {

class Connection;
class Layer;

class INFERENCE_ENGINE_API_CLASS(Port) {
public:
    Port(Builder::Network& network, PortInfo port, bool isInput);
    PortData::Ptr getData() const;
    const std::map<std::string, Parameter>& getParameters() const;
    Layer getLayer() const;
    Connection getConnection() const;
    void connect(const Port& port);
    void disconnect();
    const SizeVector& shape() const;
    PortInfo getPortInfo() const;
    bool operator==(const Port& rObj) const;
    bool operator!=(const Port& rObj) const;

private:
    Builder::Network& network;
    PortInfo port;
    bool input;

    friend class Connection;
};

class INFERENCE_ENGINE_API_CLASS(Layer) {
public:
    Layer(Builder::Network& network, idx_t id);
    Port getInPort() const;
    Port getInPort(idx_t idx) const;
    std::vector<Port> getInPorts() const;
    Port getOutPort() const;
    Port getOutPort(idx_t idx) const;
    std::vector<Port> getOutPorts() const;

    void setParameter(const std::string& key, const Parameter& value);
    Parameter& getParameter(const std::string& value) const;

    idx_t getId() const;
    std::string getName() const;
    std::string getType() const;
    operator Builder::Layer::Ptr() const;

private:
    Builder::Network& network;
    idx_t layerId;

    Builder::Layer::Ptr getLayer() const;
};

class INFERENCE_ENGINE_API_CLASS(Connection) {
public:
    explicit Connection(const Port& port);
    Connection(Builder::Network& network, const InferenceEngine::Connection& connection);
    Connection(Builder::Network& network, const PortInfo& inPort, const PortInfo& outPort);
    Connection(Builder::Network& network, const PortInfo& inPort, const std::vector<PortInfo>& outPorts);

    Port getSource() const;
    void setSource(const Port& port);
    Port getDestination() const;
    Port getDestination(idx_t idx);
    std::vector<Port> getDestinations() const;
    void addDestination(const Port& port);
    void setDestination(const Port& port);
    void setDestinations(const std::vector<Port>& ports);
    void remove();

private:
    Builder::Network& network;
    PortInfo inPort;
    std::vector<PortInfo> outPorts;

    bool inPortExist() const;
};

class INFERENCE_ENGINE_API_CLASS(Network) {
public:
    explicit Network(Builder::Network& network): network(network) {}
    virtual ~Network() = default;

    Layer addLayer(const Builder::Layer& layer);
    void removeLayer(const Layer& layer);
    Layer getLayer(const std::string& name) const;
    Layer getLayer(idx_t id) const;

    Builder::Network& getBuilderNetwork() const;

    Connection connect(const Layer& src, const Layer& dst);
    Connection connect(const Port& src, const Port& dst);
    void disconnect(const Layer& src, const Layer& dst);
    void disconnect(const Port& src, const Port& dst);
    Connection getConnection(const Layer& src, const Layer& dst) const;
    Connection getConnection(const Port& src, const Port& dst) const;

private:
    Builder::Network& network;
};

}  // namespace Transform
}  // namespace InferenceEngine
