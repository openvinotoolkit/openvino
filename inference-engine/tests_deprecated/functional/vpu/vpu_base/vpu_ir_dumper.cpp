// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ir_dumper.hpp"
#include "single_layer_common.hpp"
#include "debug.h"

namespace {
    std::string paramterPresitionToString(InferenceEngine::Precision precision) {
        switch (precision) {
        case InferenceEngine::Precision::FP16: return "f16";
        case InferenceEngine::Precision::FP32: return "f32";
        case InferenceEngine::Precision::I64: return "i64";
        default:
            break;
        }
        IE_ASSERT(false) << "Unsupported input presision type: " << precision;
        return "";
    }
}

class IRDumperEdge {
public:
    static constexpr int startingOutputPort = 10;

public:
    IRDumperEdge() = delete;
    IRDumperEdge(const IRDumperLayer * from, const IRDumperLayer * to, const size_t portFrom, const size_t portTo)
        : _from(from), _to(to), _portFrom(portFrom), _portTo(portTo) {}

public:
    const IRDumperLayer * _from = nullptr;
    const IRDumperLayer * _to   = nullptr;
    size_t                _portFrom = startingOutputPort;
    size_t                _portTo = 0;

public:
    IRXmlNode dump() const;
};

// -------------------------  IRWeights -------------------------------------------------

size_t IRWeightsDescription::size() const { return _data.size(); }

bool IRWeightsDescription::empty() const { return _data.empty(); }

InferenceEngine::SizeVector IRWeightsDescription::desc() const {
    // _desc.empty() means "autogenerate tensor description for common cases".
    // if we explicitly set weights to be scalar, then return empty description.
    // that is mostly done for symplyfying tests code which does not support IRv10.
    return _isScalar ? InferenceEngine::SizeVector{}
                     : (_desc.empty() ? InferenceEngine::SizeVector{_data.size()}
                                      : _desc);
}

size_t IRWeightsDescription::fill(uint8_t* destination, size_t offset) {
    IE_ASSERT(_data.size());
    memcpy(destination, _data.data(), _data.size());
    _dataOffset = offset;
    return _data.size();
}

// -------------------------  IRDumperNetwork -------------------------------------------------

IRDumperNetwork::IRDumperNetwork(IRVersion version) : _version(version) {}

IRDumperNetwork::~IRDumperNetwork() = default;

IRXmlNode IRDumperNetwork::dump() const {
    IRXmlNode net {"net", {
            {"batch", "1"},
            {"name" , "model.ckpt"},
            {"precision"  , "FP16"},
            {"version"   , (_version == IRVersion::v7 ? "7" : "10")},
        }, {}, {}};
    IRXmlNode layers {"layers", {}, {}, {}};
    for (const auto& layer : _layers) {
        layers.children.push_back(layer.dump());
    }
    IRXmlNode edges {"edges", {}, {}, {}};
    for (const auto& edge : _edges) {
        edges.children.push_back(edge.dump());
    }
    net.children.push_back(std::move(layers));
    net.children.push_back(std::move(edges));
    return net;
}

IRDumperLayer& IRDumperNetwork::addLayer(const std::string& name, const std::string& type, const IN_OUT_desc& in, const IN_OUT_desc& out) {
    IRDumperLayer l;
    l._version = _version;
    l._name = name;
    l._type = type;
    l._inDesc = in;
    l._outDesc = out;
    _layers.push_back(l);
    return *_layers.rbegin();
}

void IRDumperNetwork::addInput(const std::string& name, const IN_OUT_desc& out) {
    IE_ASSERT(out.size() >= 1);
    _inputLayersCount = out.size();
    if (_inputLayersCount == 1) {
        auto & l = addLayer(name, _version == IRVersion::v7 ? "Input" : "Parameter", {}, out);
        if (_version == IRVersion::v10)
            l._parameterPrecision = InferenceEngine::Precision::FP16;
    } else {
        for (size_t i = 0; i < _inputLayersCount; ++i) {
            auto & l = addLayer(name + std::to_string(i), _version == IRVersion::v7 ? "Input" : "Parameter", {}, {out[i]});
            if (_version == IRVersion::v10)
                l._parameterPrecision = InferenceEngine::Precision::FP16;
        }
    }
}

void IRDumperNetwork::addOutput(const std::string& name, const IN_OUT_desc& in) {
    if (_version == IRVersion::v10)
        addLayer(name,  "Result", in, {});
}

void IRDumperNetwork::finalize() {
    makeEdges();
    populateWeights();
    makeLayerSequence();
}

void IRDumperNetwork::makeEdges() {
    for (size_t i = 0; i < _inputLayersCount; ++i) {
        createEdge(_layers[i], _layers[_inputLayersCount], IRDumperEdge::startingOutputPort, i);
    }
    for (size_t i = _inputLayersCount; i < _layers.size() - 1; ++i) {
        createEdge(_layers[i], _layers[i + 1], IRDumperEdge::startingOutputPort, 0);
    }
}

void IRDumperNetwork::populateWeights() {
    size_t totalSize = 0;
    for (const auto& layer : _layers) {
        totalSize += layer._weights.size();
        totalSize += layer._biases.size();
        for (const auto& param : layer._paramWeights)
            totalSize += param.size();
    }
    if (!totalSize)
        return;

    uint8_t* dataPtr;
    {
        auto* w = new WeightsBlob({InferenceEngine::Precision::U8, {(totalSize)}, InferenceEngine::C});
        w->allocate(); // private
        _weights.reset(w);
        auto d = w->data();
        dataPtr = d.as<uint8_t*>();
    }

    std::vector<IRDumperLayer*> ptrs;
    for (auto& layer : _layers) {
        ptrs.push_back(&layer);
    }
    if (_version == IRVersion::v10) {
        for (auto* layerPtr : ptrs) {
            IRDumperLayer & layer = *layerPtr;
            if (!layer._weights.empty()) {
                layer._paramWeights.emplace_back(std::move(layer._weights));
            }
            for (auto&& weightsDesc : layer._paramWeights) {
                const size_t oldInSize = layer._inDesc.size();

                auto& dataLayer = addLayer(layer._name + "/" + weightsDesc._description, "Const", {}, {weightsDesc.desc()});
                layer._inDesc.push_back(weightsDesc.desc());

                dataLayer._weights = std::move(weightsDesc);
                dataLayer._outputPrecision = dataLayer._weights._precision;

                createEdge(dataLayer         , layer, IRDumperEdge::startingOutputPort, oldInSize);
            }
            layer._paramWeights.clear();

            if (!layer._biases.empty()) {
                auto& constDataLayer = addLayer(layer._name + "/biasData", "Const", {}, {layer._biases.desc()});
                auto& additionLayer  = addLayer(layer._name + "/add", "Add", {layer._outDesc[0], layer._biases.desc()}, layer._outDesc);
                constDataLayer._weights = std::move(layer._biases);

                IRDumperEdge* currentConvOutEdge = nullptr;
                for (auto& edge : _edges) {
                    if (edge._from == &layer) {
                        currentConvOutEdge = &edge;
                        break;
                    }
                }
                IE_ASSERT(currentConvOutEdge);
                currentConvOutEdge->_from = &additionLayer;

                createEdge(layer         , additionLayer, IRDumperEdge::startingOutputPort, 0);
                createEdge(constDataLayer, additionLayer, IRDumperEdge::startingOutputPort, 1);
            }
        }
    }
    size_t offset = 0;
    for (auto& layer : _layers) {
        if (!layer._weights.empty()) {
            offset += layer._weights.fill(dataPtr + offset, offset);
        }
        if (!layer._biases.empty()) {
            offset += layer._biases.fill(dataPtr + offset, offset);
        }
    }
}

void IRDumperNetwork::makeLayerSequence() {
    for (size_t i = 0; i < _layers.size(); ++i) {
        _layers[i]._id = i;
    }
}

void IRDumperNetwork::createEdge(const IRDumperLayer& from, const IRDumperLayer& to, size_t portFrom, size_t portTo) {
    _edges.emplace_back(&from, &to, portFrom, portTo);
}

// -------------------------  IRDumperLayer -------------------------------------------------

IRXmlNode IRDumperLayer::dump() const {
    IRXmlNode layer {"layer", {{"id", std::to_string(_id)}, {"name", _name}, {"type", _type}}, {}, {}};

    if (_version == IRVersion::v10) {
        layer.attributes["version"] = "opset1";
        if (!_weights.empty()) {
            IRXmlNode dataNode {"data", {
                    {"offset", std::to_string(_weights._dataOffset)},
                    {"size", std::to_string(_weights.size())},
                    {"element_type", paramterPresitionToString(_weights._precision)},
                    {"shape", InferenceEngine::details::joinVec(_outDesc[0])}}, {}, {}};
            layer.children.push_back(std::move(dataNode));
        }
        else if (_parameterPrecision != InferenceEngine::Precision::UNSPECIFIED) {
            IRXmlNode dataNode {"data", {
                    {"element_type", paramterPresitionToString(_parameterPrecision)},
                    {"shape", InferenceEngine::details::joinVec(_outDesc[0])}}, {}, {}};
            layer.children.push_back(std::move(dataNode));
        }
        else if (!_dataParams.empty()) {
            IRXmlNode dataNode {"data", _dataParams, {}, {}};
            layer.children.push_back(std::move(dataNode));
        }
    }else {
        if (!_dataParams.empty()) {
            IRXmlNode dataNode {"data", _dataParams, {}, {}};
            layer.children.push_back(std::move(dataNode));
        }
        if (!_weights.empty()) {
            IRXmlNode weights {"weights", {
                    {"offset", std::to_string(_weights._dataOffset)},
                    {"size", std::to_string(_weights.size())}}, {}, {}};
            layer.children.push_back(std::move(weights));
        }
        if (!_biases.empty()) {
            IRXmlNode biases {"biases", {
                    {"offset", std::to_string(_biases._dataOffset)},
                    {"size", std::to_string(_biases.size())}}, {}, {}};
            layer.children.push_back(std::move(biases));
        }
    }
    if (!_inDesc.empty())
        layer.children.push_back(dumpDesc(_inDesc, "input", 0, InferenceEngine::Precision::UNSPECIFIED));
    if (!_outDesc.empty())
        layer.children.push_back(dumpDesc(_outDesc, "output", IRDumperEdge::startingOutputPort, _outputPrecision));

    return layer;
}

IRXmlNode IRDumperLayer::dumpDesc(const IN_OUT_desc& desc, const std::string& portsTag, int portIndexStart, const InferenceEngine::Precision& precision) const {
    IRXmlNode ports {portsTag, {}, {}, {}};

    int portIndex = portIndexStart;
    for (const auto& portDims : desc) {
        IRXmlNode port {"port", {{"id", std::to_string(portIndex++)}}, {}, {}};
        if (precision != InferenceEngine::Precision::UNSPECIFIED) {
            port.attributes["precision"] = precision.name();
        }
        for (const auto& dim : portDims) {
            IRXmlNode dimNode {"dim", {}, std::to_string(dim), {}};
            port.children.push_back(std::move(dimNode));
        }
        ports.children.push_back(std::move(port));
    }
    return ports;
}

// -------------------------  IRDumperEdge -------------------------------------------------

IRXmlNode IRDumperEdge::dump() const {
    IRXmlNode egde {"edge", {
            {"from-layer", std::to_string(_from->id())},
            {"from-port" , std::to_string(_portFrom)},
            {"to-layer"  , std::to_string(_to->id())},
            {"to-port"   , std::to_string(_portTo)},
        }, {}, {}};
    return egde;
}

// -------------------------  common utils -------------------------------------------------

std::string formatXmlNode(const IRXmlNode& node, int indent) {
    std::ostringstream os;
    os << std::string(indent, '\t') << "<" << node.name;
    for (const auto& pair : node.attributes)
        os << " " << pair.first + "=\"" + pair.second + "\"";
    if (node.rawText.empty() && node.children.empty()) {
        os << "/>\n";
        return os.str();
    }
    os << ">";

    if (!node.rawText.empty()) {
        os << node.rawText << "</" << node.name << ">\n";
        return os.str();
    }
    os << "\n";
    for (const auto& child : node.children)
        os << formatXmlNode(child, indent + 1);

    os << std::string(indent, '\t') << "</" << node.name << ">\n";
    return os.str();
}
