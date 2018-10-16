// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <memory>
#include "cnn_network_impl.hpp"
#include "ie_layers.h"
#include "parsers.h"
#include "caseless.hpp"
#include <vector>

namespace InferenceEngine {
namespace details {
struct WeightSegment {
    Precision precision;
    // offset in bytes of the global weights array
    size_t start = 0;
    // size in bytes
    size_t size = 0;

    inline size_t getEnd() const { return start + size; }

    // checks if this segment is in the range of 0 to rangeSize, safer than using getEnd() to avoid int overflow
    inline bool inRange(size_t rangeSize) const {
        return start < rangeSize && (rangeSize - start) >= size;
    }
};

struct LayerParseParameters {
    struct LayerPortData {
        int           portId;
        Precision     precision;
        SizeVector    dims;
    };
    InferenceEngine::LayerParams prms;
    int layerId = -1;
    std::vector<LayerPortData> inputPorts;
    std::vector<LayerPortData> outputPorts;
    std::map<std::string, WeightSegment> blobs;

    void addOutputPort(const LayerPortData &port);
    void addInputPort(const LayerPortData &port);
};

class BaseCreator {
    std::string type_;
protected:
    explicit BaseCreator(const std::string& type) : type_(type) {}

public:
    virtual ~BaseCreator() {}
    static int version_;

    virtual CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) = 0;

    bool shouldCreate(const std::string& nodeType) const {
        CaselessEq<std::string> comparator;
        return comparator(nodeType, type_);
    }
};

class V2FormatParser : public IFormatParser {
public:
    explicit V2FormatParser(int version);

    CNNNetworkImplPtr Parse(pugi::xml_node& root) override;

    Blob::Ptr GetBlobFromSegment(const TBlob<uint8_t>::Ptr& weights, const WeightSegment & weight_segment) const;
    void SetWeights(const TBlob<uint8_t>::Ptr& weights) override;
    void ParseDims(SizeVector& dims, const pugi::xml_node &node) const;

private:
    int _version;
    Precision _defPrecision;
    std::map<std::string, LayerParseParameters> layersParseInfo;
    std::map<std::string, DataPtr> _portsToData;

    CNNNetworkImplPtr _network;
    std::map<std::string, std::vector<WeightSegment>> _preProcessSegments;
    const std::vector<std::shared_ptr<BaseCreator> > &getCreators() const;
    void ParsePort(LayerParseParameters::LayerPortData& port, pugi::xml_node &node) const;
    void ParseGenericParams(pugi::xml_node& node, LayerParseParameters& layerParsePrms) const;
    CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& prms) const;

    void SetLayerInput(CNNNetworkImpl& network, const std::string& data, CNNLayerPtr& targetLayer, int inputPort);

    DataPtr ParseInputData(pugi::xml_node& root) const;

    void ParsePreProcess(pugi::xml_node& node);
};
}  // namespace details
}  // namespace InferenceEngine
