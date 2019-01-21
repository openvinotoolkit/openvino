// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ie_blob.h>

#include <iterator>

#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

#include <sstream>
#include <ie_builders.hpp>
#include <inference_engine.hpp>

#include <ie_api_impl.hpp>


// namespace IE Python
namespace InferenceEnginePython {
struct LayerBuilder;

struct Port {
    Port() = default;

    explicit Port(const std::vector<size_t> &shapes);

    InferenceEngine::Port actual;
    std::vector<size_t> shape;
};

struct ILayer {
    InferenceEngine::ILayer::CPtr layer_ptr;
    std::string name;
    size_t id;
    std::string type;
    std::map<std::string, std::string> parameters;
    std::map<std::string, InferenceEngine::Blob::Ptr> constant_data;
    std::vector<Port> in_ports;
    std::vector<Port> out_ports;
};

struct PortInfo {
    PortInfo(size_t layer_id, size_t port_id);

    PortInfo() : actual(0, 0) {}

    InferenceEngine::PortInfo actual;
    size_t layer_id;
    size_t port_id;
};

struct Connection {
    Connection() : actual(InferenceEngine::PortInfo(0), InferenceEngine::PortInfo(0)) {}

    Connection(PortInfo input, PortInfo output);

    InferenceEngine::Connection actual;
    PortInfo _from;
    PortInfo to;
};

struct INetwork {
    InferenceEngine::INetwork::Ptr actual;
    std::string name;
    size_t size;
    std::vector<ILayer> layers;
    std::vector<ILayer> inputs;
    std::vector<ILayer> outputs;

    std::vector<Connection> getLayerConnections(size_t layer_id);

    IENetwork to_ie_network();
};

struct NetworkBuilder {
    InferenceEngine::Builder::Network::Ptr network_ptr;

    explicit NetworkBuilder(const std::string &name);

    NetworkBuilder() = default;

    NetworkBuilder from_ie_network(const InferenceEnginePython::IENetwork &icnn_net);

    INetwork build();

    std::vector<LayerBuilder> getLayers();

    LayerBuilder getLayer(size_t layer_id);

    void removeLayer(const LayerBuilder &layer);

    size_t addLayer(const LayerBuilder &layer);

    size_t addAndConnectLayer(const std::vector<PortInfo> &input, const LayerBuilder &layer);

    const std::vector<Connection> getLayerConnections(const LayerBuilder &layer);

    void disconnect(const Connection &connection);

    void connect(const PortInfo &input, const PortInfo &output);
};

struct LayerBuilder {
    InferenceEngine::Builder::Layer actual;
    size_t id;

    LayerBuilder(const std::string &type, const std::string &name);

    LayerBuilder() : actual("", "") {}

    LayerBuilder from_ilayer(const ILayer &ilayer);

    const std::string &getName();

    void setName(const std::string &name);

    const std::string &getType();

    void setType(const std::string &type);

    std::vector<Port> getInputPorts();

    void setInputPorts(const std::vector<Port> ports);

    std::vector<Port> getOutputPorts();

    void setOutputPorts(const std::vector<Port> ports);


    std::map<std::string, std::string> getParameters();

    void setParameters(std::map<std::string, std::string> params_map);

    ILayer build();

    std::map<std::string, InferenceEngine::Blob::Ptr> getConstantData();

    InferenceEngine::Blob::Ptr allocateBlob(std::vector<size_t> dims, const std::string &precision);

    void setConstantData(const std::map<std::string, InferenceEngine::Blob::Ptr> &const_data);

// TODO(  ): Fix LAyerBuilder object copying - pass by reference
//    void addConstantData(const std::string & name, InferenceEngine::Blob::Ptr data);
};
}  // namespace InferenceEnginePython
