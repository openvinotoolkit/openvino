// Copyright (C) 2018-2019 Intel Corporation
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

#include <cpp/ie_cnn_network.h>
#include <details/caseless.hpp>

#include <vpu/frontend/stage_builder.hpp>
#include <vpu/frontend/parse_network.hpp>
#include <vpu/model/model.hpp>
#include <vpu/custom_layer.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;


class FrontEnd final : public std::enable_shared_from_this<FrontEnd> {
//
// Public API
//

public:
    using Ptr = std::shared_ptr<FrontEnd>;

    explicit FrontEnd(const StageBuilder::Ptr& stageBuilder) : _stageBuilder(stageBuilder) {}

    Model::Ptr buildInitialModel(const ie::ICNNNetwork& network);

    std::set<std::string> checkSupportedLayers(const ie::ICNNNetwork& network);

    const std::vector<ie::CNNLayerPtr>& allLayers() const { return _ieNetworkParser.orderedLayers; }

//
// Passes
//

private:
    Model::Ptr runCommonPasses(
            const ie::ICNNNetwork& network,
            LayersOrder order);

    ie::CNNNetwork detectNetworkBatch(
            const ie::ICNNNetwork& network,
            const Model::Ptr& model);

    void RemoveConstLayers(ie::ICNNNetwork& network);

    void parseInputAndOutputData(const Model::Ptr& model);
    void addDataTypeConvertStages(const Model::Ptr& model);
    void addPreProcessStages(const Model::Ptr& model);

    void eliminatePriorBoxData(const Model::Ptr& model);

//
// IR Parsers
//

public:
    //
    // Layers, that might be both SW and HW
    //

    void parseConvolution(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePooling(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseFullyConnected(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);

    //
    // SW only layers
    //

    void parseReLU(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseSoftMax(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseGRN(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseMVN(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseNorm(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePower(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseScale(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePermute(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseDetectionOutput(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseEltwise(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseSigmoid(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseTanH(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePReLU(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseBatchNorm(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseDeconvolution(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseCopy(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseELU(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseCrop(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseTile(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseNormalize(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseRegionYolo(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseReorgYolo(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseBias(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseCTCDecoder(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseInterp(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseClamp(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseProposal(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseROIPooling(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePSROIPooling(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseCustom(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseMTCNN(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseLSTMCell(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePad(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseResample(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseArgMax(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseRNN(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);

    //
    // Special layers
    //

    void parsePriorBox(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parsePriorBoxClustered(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseReshape(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseConcat(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);
    void parseSplit(const Model::Ptr& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs);

//
// Utility
//

private:
    Data getVpuData(const ie::DataPtr& ieData);
    void bindData(const Data& data, const ie::DataPtr& ieData);

    void getInputAndOutputData(
            const Model::Ptr& model,
            const ie::CNNLayerPtr& layer,
            DataVector& inputs,
            DataVector& outputs);

    std::tuple<Data, Data> getWeightsAndBiases(
            const Model::Ptr& model,
            const ie::CNNLayerPtr& layer);

//
// Internal state
//

private:
    StageBuilder::Ptr _stageBuilder;

    std::unordered_set<ie::DataPtr> _unbatchedOutputs;
    std::unordered_map<ie::DataPtr, Data> _ieToVpuMap;

    ie::details::caseless_map<std::string, CustomLayer::Ptr> _customLayers;
    vpu::IeNetworkParser _ieNetworkParser;
};

}  // namespace vpu
