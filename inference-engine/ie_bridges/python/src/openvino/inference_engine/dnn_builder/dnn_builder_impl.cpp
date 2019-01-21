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

#include "dnn_builder_impl.hpp"

// using namespace InferenceEnginePython;
// using namespace std;

std::map<std::string, InferenceEngine::Precision> precision_map = {{"FP32", InferenceEngine::Precision::FP32},
                                                                   {"FP16", InferenceEngine::Precision::FP16},
                                                                   {"Q78",  InferenceEngine::Precision::Q78},
                                                                   {"I32",  InferenceEngine::Precision::I32},
                                                                   {"I16",  InferenceEngine::Precision::I16},
                                                                   {"I8",   InferenceEngine::Precision::I8},
                                                                   {"U16",  InferenceEngine::Precision::U16},
                                                                   {"U8",   InferenceEngine::Precision::U8}};

InferenceEnginePython::ILayer buildILayer(InferenceEngine::ILayer::CPtr it) {
    std::vector<InferenceEnginePython::Port> in_ports;
    std::vector<InferenceEnginePython::Port> out_ports;
    for (const auto &port : it->getInputPorts()) {
        in_ports.push_back(InferenceEnginePython::Port(port.shape()));
    }
    for (const auto &port : it->getOutputPorts()) {
        out_ports.push_back(InferenceEnginePython::Port(port.shape()));
    }

    std::map<std::string, std::string> params_map;
    for (const auto &params : it->getParameters()->getParameters()) {
        params_map.emplace(params.first, params.second);
    }
    std::map<std::string, InferenceEngine::Blob::Ptr> data_map;
    for (const auto &data : it->getParameters()->getConstantData()) {
        data_map.emplace(data.first, std::const_pointer_cast<InferenceEngine::Blob>(data.second));
    }
    return {it,
            it->getName(),
            it->getId(),
            it->getType(),
            params_map,
            data_map,
            in_ports,
            out_ports,
    };
}

// NetworkBuilder
InferenceEnginePython::NetworkBuilder::NetworkBuilder(const std::string &name) {
    // TODO(  ): std::move or instance in heap? Please check in other places.
    InferenceEngine::Builder::Network network(name);
    network_ptr = std::make_shared<InferenceEngine::Builder::Network>(network);
}

InferenceEnginePython::NetworkBuilder InferenceEnginePython::NetworkBuilder::from_ie_network(
        const InferenceEnginePython::IENetwork &icnn_net) {
    InferenceEngine::Builder::Network network((InferenceEngine::ICNNNetwork &) icnn_net.actual);
    NetworkBuilder net_builder = NetworkBuilder();
    net_builder.network_ptr = std::make_shared<InferenceEngine::Builder::Network>(network);
    return net_builder;
}

InferenceEnginePython::INetwork InferenceEnginePython::NetworkBuilder::build() {
    InferenceEngine::INetwork::Ptr i_net = network_ptr->build();
    std::vector<ILayer> layers;
    for (const auto &it : *i_net) {
        layers.push_back(buildILayer(it));
    }
    std::vector<ILayer> inputs;
    for (const auto &it : i_net->getInputs()) {
        inputs.push_back(buildILayer(it));
    }
    std::vector<ILayer> outputs;
    for (const auto &it : i_net->getInputs()) {
        outputs.push_back(buildILayer(it));
    }
    return {i_net,             // INetwork ptr
            i_net->getName(),  // name
            i_net->size(),     // Number of layers
            layers,
            inputs,
            outputs
    };
}

std::vector<InferenceEnginePython::LayerBuilder> InferenceEnginePython::NetworkBuilder::getLayers() {
    std::vector<LayerBuilder> layers;
    for (const auto &it : network_ptr->getLayers()) {
        LayerBuilder layer;
        layer.actual = it;
        layer.id = it.getId();
        layers.push_back(layer);
    }
    return layers;
}

InferenceEnginePython::LayerBuilder InferenceEnginePython::NetworkBuilder::getLayer(size_t layer_id) {
    LayerBuilder layer;
    InferenceEngine::Builder::Layer ie_layer = network_ptr->getLayer(layer_id);
    layer.actual = ie_layer;
    layer.id = ie_layer.getId();
    return layer;
}

void InferenceEnginePython::NetworkBuilder::removeLayer(const LayerBuilder &layer) {
    network_ptr->removeLayer(layer.id);
}

const std::vector<InferenceEnginePython::Connection> InferenceEnginePython::NetworkBuilder::getLayerConnections(
        const LayerBuilder &layer) {
    std::vector<InferenceEngine::Connection> ie_connections = network_ptr->getLayerConnections(layer.id);
    std::vector<Connection> connections;
    for (auto const &it : ie_connections) {
        PortInfo input(it.from().layerId(), it.from().portId());
        PortInfo output(it.to().layerId(), it.to().portId());
        connections.push_back(Connection(input, output));
    }
    return connections;
}

void InferenceEnginePython::NetworkBuilder::disconnect(const Connection &connection) {
    network_ptr->disconnect(connection.actual);
}

void InferenceEnginePython::NetworkBuilder::connect(const PortInfo &input, const PortInfo &output) {
    network_ptr->connect(input.actual, output.actual);
}

size_t InferenceEnginePython::NetworkBuilder::addLayer(const LayerBuilder &layer) {
    return network_ptr->addLayer(layer.actual);
}

size_t InferenceEnginePython::NetworkBuilder::addAndConnectLayer(const std::vector<PortInfo> &input,
                                                                 const LayerBuilder &layer) {
    std::vector<InferenceEngine::PortInfo> ie_ports;
    for (const auto &it : input) {
        ie_ports.push_back(it.actual);
    }
    return network_ptr->addLayer(ie_ports, layer.actual);
}
// NetworkBuilder end
// NetworkBuilder end

// Port
InferenceEnginePython::Port::Port(const std::vector<size_t> &shapes) {
    actual = InferenceEngine::Port(shapes);
    shape = actual.shape();
}

InferenceEnginePython::PortInfo::PortInfo(size_t layer_id, size_t port_id) : PortInfo() {
    this->actual = InferenceEngine::PortInfo(layer_id, port_id);
    this->layer_id = layer_id;
    this->port_id = port_id;
}
// Port end

// INetwork
std::vector<InferenceEnginePython::Connection> InferenceEnginePython::INetwork::getLayerConnections(size_t layer_id) {
    std::vector<Connection> connections;
    for (const auto &it : actual->getLayerConnections(layer_id)) {
        PortInfo input = PortInfo(it.from().layerId(), it.from().portId());
        PortInfo output = PortInfo(it.to().layerId(), it.to().portId());
        connections.push_back(Connection(input, output));
    }
    return connections;
}

InferenceEnginePython::IENetwork InferenceEnginePython::INetwork::to_ie_network() {
    std::shared_ptr<InferenceEngine::ICNNNetwork> icnn_net = InferenceEngine::Builder::convertToICNNNetwork(actual);
    InferenceEngine::CNNNetwork cnn_net(icnn_net);
    IENetwork ie_net = IENetwork();
    ie_net.actual = cnn_net;
    ie_net.name = name;
    ie_net.batch_size = cnn_net.getBatchSize();
    return ie_net;
}
// INetwork end

// Connection
InferenceEnginePython::Connection::Connection(PortInfo input, PortInfo output) : Connection() {
    this->actual = InferenceEngine::Connection(InferenceEngine::PortInfo(input.layer_id, input.port_id),
                                               InferenceEngine::PortInfo(output.layer_id, output.port_id));
    this->_from = PortInfo(actual.from().layerId(), actual.from().portId());
    this->to = PortInfo(actual.to().layerId(), actual.to().portId());
}
// Connection end

// LayerBuilder
InferenceEnginePython::LayerBuilder::LayerBuilder(const std::string &type, const std::string &name) : LayerBuilder() {
    InferenceEngine::Builder::Layer layer(type, name);
    this->actual = layer;
    this->id = layer.getId();
}

const std::string &InferenceEnginePython::LayerBuilder::getName() {
    return actual.getName();
}

const std::string &InferenceEnginePython::LayerBuilder::getType() {
    return actual.getType();
}

std::vector<InferenceEnginePython::Port> InferenceEnginePython::LayerBuilder::getInputPorts() {
    std::vector<Port> ports;
    for (const auto &it : actual.getInputPorts()) {
        ports.push_back(Port(it.shape()));
    }
    return ports;
}

std::vector<InferenceEnginePython::Port> InferenceEnginePython::LayerBuilder::getOutputPorts() {
    std::vector<Port> ports;
    for (const auto &it : actual.getOutputPorts()) {
        ports.push_back(Port(it.shape()));
    }
    return ports;
}

std::map<std::string, std::string> InferenceEnginePython::LayerBuilder::getParameters() {
    std::map<std::string, std::string> params_map;
    for (const auto &it : actual.getParameters()) {
        params_map.emplace(it.first, it.second);
    }
    return params_map;
}

void InferenceEnginePython::LayerBuilder::setParameters(std::map<std::string, std::string> params_map) {
    std::map<std::string, InferenceEngine::Parameter> ie_params_map;
    for (const auto &it : params_map) {
        InferenceEngine::Parameter ie_param((it.second));
        ie_params_map.emplace(it.first, ie_param);
    }
    actual = actual.setParameters(ie_params_map);
}

void InferenceEnginePython::LayerBuilder::setName(const std::string &name) {
    actual = actual.setName(name);
}

void InferenceEnginePython::LayerBuilder::setType(const std::string &type) {
    actual = actual.setType(type);
}

void InferenceEnginePython::LayerBuilder::setInputPorts(const std::vector<Port> ports) {
    std::vector<InferenceEngine::Port> ie_ports;
    for (const auto &it : ports) {
        ie_ports.push_back(it.actual);
    }
    actual = actual.setInputPorts(ie_ports);
}

void InferenceEnginePython::LayerBuilder::setOutputPorts(const std::vector<Port> ports) {
    std::vector<InferenceEngine::Port> ie_ports;
    for (const auto &it : ports) {
        ie_ports.push_back(it.actual);
    }
    actual = actual.setOutputPorts(ie_ports);
}

InferenceEnginePython::ILayer InferenceEnginePython::LayerBuilder::build() {
    return buildILayer(actual.build());
}

std::map<std::string, InferenceEngine::Blob::Ptr> InferenceEnginePython::LayerBuilder::getConstantData() {
    std::map<std::string, InferenceEngine::Blob::Ptr> data_map;
    for (const auto &it : actual.getConstantData()) {
        data_map.emplace(it.first, std::const_pointer_cast<InferenceEngine::Blob>(it.second));
    }
    return data_map;
}

InferenceEngine::Blob::Ptr InferenceEnginePython::LayerBuilder::allocateBlob(std::vector<size_t> dims,
                                                                             const std::string &precision) {
    InferenceEngine::Layout ie_layout;
    ie_layout = InferenceEngine::TensorDesc::getLayoutByDims(dims);
    InferenceEngine::Precision ie_precision = precision_map.at(precision);
    const InferenceEngine::TensorDesc &tdesc = InferenceEngine::TensorDesc(ie_precision, dims, ie_layout);
    InferenceEngine::Blob::Ptr blob;
    switch (ie_precision) {
        case InferenceEngine::Precision::FP32:
            blob = InferenceEngine::make_shared_blob<float>(tdesc);
            break;
        case InferenceEngine::Precision::FP16:
            blob = InferenceEngine::make_shared_blob<int>(tdesc);
            break;
        case InferenceEngine::Precision::I16:
            blob = InferenceEngine::make_shared_blob<int>(tdesc);
            break;
        case InferenceEngine::Precision::U16:
            blob = InferenceEngine::make_shared_blob<int>(tdesc);
            break;
        case InferenceEngine::Precision::U8:
            blob = InferenceEngine::make_shared_blob<unsigned char>(tdesc);
            break;
        case InferenceEngine::Precision::I8:
            blob = InferenceEngine::make_shared_blob<signed char>(tdesc);
            break;
        case InferenceEngine::Precision::I32:
            blob = InferenceEngine::make_shared_blob<signed int>(tdesc);
            break;
        default:
            blob = InferenceEngine::make_shared_blob<float>(tdesc);
            break;
    }

    blob->allocate();
    return blob;
}

void InferenceEnginePython::LayerBuilder::setConstantData(const std::map<std::string,
                                                          InferenceEngine::Blob::Ptr> &const_data) {
    actual.setConstantData(const_data);
}
// TODO(  ): Fix LAyerBuilder object copying - pass by reference
// void LayerBuilder::addConstantData(const std::string & name, InferenceEngine::Blob::Ptr data){
//     InferenceEngine::Blob::CPtr c_data = const_pointer_cast<const InferenceEngine::Blob>(data);
//     actual.addConstantData(name, c_data);
// }

// LayerBuilder end
