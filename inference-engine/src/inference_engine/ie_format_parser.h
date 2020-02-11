// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cnn_network_impl.hpp"
#include "details/caseless.hpp"
#include "ie_layers.h"
#include "parsers.h"

namespace InferenceEngine {
namespace details {
struct WeightSegment {
    Precision precision;
    // offset in bytes of the global weights array
    size_t start = 0;
    // size in bytes
    size_t size = 0;

    inline size_t getEnd() const {
        return start + size;
    }

    // checks if this segment is in the range of 0 to rangeSize, safer than using getEnd() to avoid int overflow
    inline bool inRange(size_t rangeSize) const {
        return start < rangeSize && (rangeSize - start) >= size;
    }
};

struct LayerParseParameters {
    struct LayerPortData {
        int portId;
        Precision precision;
        SizeVector dims;
    };
    InferenceEngine::LayerParams prms;
    int layerId = -1;
    std::vector<LayerPortData> inputPorts;
    std::vector<LayerPortData> outputPorts;
    std::map<std::string, WeightSegment> blobs;

    std::function<void(const TBlob<uint8_t>::Ptr& weights)> internalWeightSet;

    size_t underIRVersion = 0;

    void addOutputPort(const LayerPortData& port);
    void addInputPort(const LayerPortData& port);
};

class BaseCreator {
private:
    std::string type_;

protected:
    explicit BaseCreator(const std::string& type): type_(type) {}

public:
    virtual ~BaseCreator() {}

    virtual CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& layerParsePrms) = 0;

    bool shouldCreate(const std::string& nodeType) const {
        InferenceEngine::details::CaselessEq<std::string> comparator;
        return comparator(nodeType, type_);
    }
};

#ifdef ENABLE_IR_READER
class INFERENCE_ENGINE_API_CLASS(FormatParser): public IFormatParser {
#else
class FormatParser : public IFormatParser {
#endif
public:
    explicit FormatParser(size_t version);

    CNNNetworkImplPtr Parse(pugi::xml_node& root) override;

    Blob::Ptr GetBlobFromSegment(const TBlob<uint8_t>::Ptr& weights, const WeightSegment& weight_segment) const;
    void SetWeights(const TBlob<uint8_t>::Ptr& weights) override;
    void ParseDims(SizeVector& dims, const pugi::xml_node& node) const;
    const DataPtr& GetDataBy(int layer_id, int port_id) const;

protected:
    std::map<std::string, LayerParseParameters> layersParseInfo;

private:
    size_t _version;
    Precision _defPrecision;
    std::vector<std::shared_ptr<BaseCreator>> creators;
    std::map<std::string, DataPtr> _portsToData;

    CNNNetworkImplPtr _network;
    std::map<std::string, std::vector<WeightSegment>> _preProcessSegments;
    void ParsePort(LayerParseParameters::LayerPortData& port, pugi::xml_node& node) const;
    void ParseGenericParams(pugi::xml_node& node, LayerParseParameters& layerParsePrms) const;
    CNNLayer::Ptr CreateLayer(pugi::xml_node& node, LayerParseParameters& prms) const;

    void SetLayerInput(CNNNetworkImpl& network, const std::string& data, CNNLayerPtr& targetLayer, int inputPort);

    DataPtr ParseInputData(pugi::xml_node& root) const;

    void ParsePreProcess(pugi::xml_node& node);
    void ParseStatisticSection(const pugi::xml_node& statNode);

    // Generate different set of creators depending on required IR version
    static std::vector<std::shared_ptr<BaseCreator>> generateCreators(int version);
};
}  // namespace details
}  // namespace InferenceEngine
