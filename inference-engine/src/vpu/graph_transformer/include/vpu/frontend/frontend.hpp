// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <set>

#include <ie_icore.hpp>
#include <cpp/ie_cnn_network.h>

#include <vpu/stage_builder.hpp>
#include <vpu/frontend/ie_parsed_network.hpp>
#include <vpu/frontend/custom_layer.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/func_ref.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class FrontEnd final {
public:
    using Ptr = std::shared_ptr<FrontEnd>;

    explicit FrontEnd(StageBuilder::Ptr stageBuilder, const std::shared_ptr<ie::ICore> core);

    ModelPtr buildInitialModel(const ie::CNNNetwork& network);

    std::set<std::string> checkSupportedLayers(const ie::CNNNetwork& network);

    const std::vector<ie::CNNLayerPtr>& origLayers() const {
        return _ieParsedNetwork.orderedLayers;
    }

//
// Passes
//

private:
    ModelPtr runCommonPasses(const ie::CNNNetwork& network);

    using SupportedLayerCallback = std::function<void(const ie::CNNLayerPtr&)>;
    using UnsupportedLayerCallback = std::function<void(const Model&, const ie::CNNLayerPtr&, const DataVector&, const DataVector&, const std::string&)>;
    ModelPtr runCommonPasses(ie::CNNNetwork network, const UnsupportedLayerCallback& unsupportedLayer,
                             const SupportedLayerCallback& supportedLayer = nullptr);

    //
    // Update IE Network
    //

    void unrollLoops(
            ie::CNNNetwork& network);

    void detectNetworkBatch(
            ie::CNNNetwork& network,
            const Model& model);

    void removeConstLayers(
            ie::CNNNetwork& network);

    //
    // Process internal VPU Model
    //

    void parseInputAndOutputData(const Model& model);

    void addDataTypeConvertStages(const Model& model);

    void addPreProcessStages(const Model& model);

//
// IR Parsers
//

public:
    //
    // Layers that might be both SW and HW
    //

    void parseConvolution(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePooling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseFullyConnected(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;

    //
    // SW only layers
    //

    void parseReLU(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSoftMax(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseGRN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseMVN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseNorm(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePower(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseScale(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePermute(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseDetectionOutput(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseEltwise(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSigmoid(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseTanH(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePReLU(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseBatchNorm(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseDeconvolution(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseCopy(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseELU(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseCrop(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseTile(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseNormalize(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseRegionYolo(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseReorgYolo(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseBias(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseCTCDecoder(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseInterp(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseClamp(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseProposal(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseROIPooling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePSROIPooling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseMTCNN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePad(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseResample(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseInterpolate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseRNN(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseGEMM(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseLog(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseExp(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseReverseSequence(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseGather(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseReduce(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseFloor(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseTopK(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSelect(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseExpDetectionOutput(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseROIFeatureExtractor(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseConvert(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseErf(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseOneHot(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseExpPriorGridGenerator(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseExpGenerateProposals(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseScatterUpdate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseScatterElementsUpdate(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseExpTopKROIs(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseNonZero(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseROIAlign(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseOutShapeOfReshape(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseBroadcast(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseStaticShapeNMS(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseMish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseGelu(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSoftPlus(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSwish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseActivation(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseLogicalNot(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseGatherND(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseHSwish(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseCeiling(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseRound(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseCTCGreedyDecoderSeqLen(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseAbs(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;

    //
    // Special layers
    //

    void parsePriorBox(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parsePriorBoxClustered(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseReshape(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseConcat(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseSplit(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseStridedSlice(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const;
    void parseDSR(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseGatherElements(const Model &model, const ie::CNNLayerPtr &layer, const DataVector &inputs, const DataVector &outputs) const;

    //
    // Parser with data sharing
    //

    void parseCustom(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseLSTMCell(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseTensorIterator(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);

    //
    // Utility
    //

    static CustomLayer::Ptr getSuitableCustomLayer(const std::vector<CustomLayer::Ptr>& customLayers, const ie::CNNLayerPtr&cnnLayer);
    static ie::CNNNetwork convertNetwork(ie::CNNNetwork& network);

private:
    Data getVpuData(const ie::DataPtr& ieData) const;
    void bindData(const Data& data, const ie::DataPtr& ieData);

    void getInputAndOutputData(
            const Model& model,
            const ie::CNNLayerPtr& layer,
            DataVector& inputs,
            DataVector& outputs);

    std::tuple<Data, Data> getWeightsAndBiases(const Model& model, const ie::CNNLayerPtr& layer) const;

    void defaultOnUnsupportedLayerCallback(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                                           const std::string& extraMessage);

    void parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                    const UnsupportedLayerCallback& onUnsupported, const SupportedLayerCallback& onSupported = nullptr);

    void processTrivialCases(const Model& model);

private:
    StageBuilder::Ptr _stageBuilder;
    const std::shared_ptr<ie::ICore> _core = nullptr;

    IeParsedNetwork _ieParsedNetwork;
    std::unordered_set<ie::DataPtr> _unbatchedOutputs;
    ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> _customLayers;

#define LAYER_PARSER(functor_name)                                                                                \
    [this](const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) \
        { functor_name(model, layer, inputs, outputs); }

    using LayerParser = std::function<void(const Model&, const ie::CNNLayerPtr&, const DataVector&, const DataVector&)>;
    const ie::details::caseless_map<std::string, LayerParser> parsers;

    std::unordered_map<ie::DataPtr, Data> _ieToVpuMap;
    ie::details::caseless_map<std::string, Data> _kernelNodes;
    std::unordered_map<ie::Blob::Ptr, Data> _lstmWeights;
    std::unordered_map<ie::Blob::Ptr, Data> _lstmBiases;
};

}  // namespace vpu
