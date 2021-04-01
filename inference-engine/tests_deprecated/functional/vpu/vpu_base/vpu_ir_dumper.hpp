// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>

#include "vpu_test_common_definitions.hpp"

class IRDumperEdge;

class IRWeightsDescription {
public:
    std::vector<uint8_t>        _data;
    InferenceEngine::Precision  _precision = InferenceEngine::Precision::FP16;
    InferenceEngine::SizeVector _desc;
    size_t                      _dataOffset = 0;
    std::string                 _description = "data";
    bool                        _isScalar = false;

public:
    size_t size() const;
    bool empty() const;
    InferenceEngine::SizeVector desc() const;
    size_t fill(uint8_t* destination, size_t offset);
};

struct IRXmlNode {
    std::string name;
    std::map<std::string, std::string> attributes;
    std::string rawText;
    std::vector<IRXmlNode> children;
};

class IRDumperLayer {    
public:
    std::string _name;
    std::string _type;
    IN_OUT_desc _inDesc;
    IN_OUT_desc _outDesc;

    InferenceEngine::Precision _parameterPrecision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision _outputPrecision = InferenceEngine::Precision::FP16;

    IRWeightsDescription _weights;
    IRWeightsDescription _biases;
    std::vector<IRWeightsDescription> _paramWeights;

    std::map<std::string, std::string> _dataParams;

public:
    IRXmlNode dump() const;
    IRXmlNode dumpDesc(const IN_OUT_desc& desc, const std::string& portsTag, int portIndexStart, const InferenceEngine::Precision& precision) const;
    size_t id() const { return _id; }

private:
    size_t _id = 0;
    IRVersion _version = IRVersion::v7;

private:
    friend class IRDumperNetwork;
};

class IRDumperNetwork {
public:
    IRDumperNetwork(IRVersion version);
    ~IRDumperNetwork();

    IRXmlNode dump() const;

    IRDumperLayer& addLayer(const std::string& name,
                  const std::string& type,
                  const IN_OUT_desc& in,
                  const IN_OUT_desc& out);
    void addInput(const std::string& name,
                  const IN_OUT_desc& out);
    void addOutput(const std::string& name,
                   const IN_OUT_desc& in);
    void finalize();

    WeightsBlob::Ptr getWeights() const {return _weights;}

private:
    void makeEdges();
    void populateWeights();
    void makeLayerSequence();

    void createEdge(const IRDumperLayer& from, const IRDumperLayer& to, size_t portFrom, size_t portTo = 0);

private:
    IRVersion                             _version;
    size_t                                _inputLayersCount = 0;
    std::deque<IRDumperLayer>             _layers;   //!< deque used for stable pointers in edges.
    std::vector<IRDumperEdge>             _edges;
    WeightsBlob::Ptr  _weights;
};

std::string formatXmlNode(const IRXmlNode& node, int indent = 0);
