// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/utils/profiling.hpp"
#include "vpu/compile_env.hpp"

#include "net_pass.h"

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <map>
#include <vector>
#include <utility>

namespace vpu {

#define LAYER_PARSER(functor_name)                                                                                \
    [this](const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) \
        { functor_name(model, layer, inputs, outputs); }

FrontEnd::FrontEnd(StageBuilder::Ptr stageBuilder)
    : _stageBuilder(std::move(stageBuilder))
    , parsers{{
        {"Convolution",                                        LAYER_PARSER(parseConvolution)},
        {"Pooling",                                            LAYER_PARSER(parsePooling)},
        {"ReLU",                                               LAYER_PARSER(parseReLU)},
        {"Clamp",                                              LAYER_PARSER(parseClamp)},
        {"FullyConnected",                                     LAYER_PARSER(parseFullyConnected)},
        {"SoftMax",                                            LAYER_PARSER(parseSoftMax)},
        {"GRN",                                                LAYER_PARSER(parseGRN)},
        {"MVN",                                                LAYER_PARSER(parseMVN)},
        {"Norm",                                               LAYER_PARSER(parseNorm)},
        {"Concat",                                             LAYER_PARSER(parseConcat)},
        {"Eltwise",                                            LAYER_PARSER(parseEltwise)},
        // Slice is represented as Split in VPU model
        {"Split",                                              LAYER_PARSER(parseSplit)},
        {"Slice",                                              LAYER_PARSER(parseSplit)},
        {"Sigmoid",                                            LAYER_PARSER(parseSigmoid)},
        {"TanH",                                               LAYER_PARSER(parseTanH)},
        {"PReLU",                                              LAYER_PARSER(parsePReLU)},
        {"Bias",                                               LAYER_PARSER(parseBias)},
        {"BatchNormalization",                                 LAYER_PARSER(parseBatchNorm)},
        {"ScaleShift",                                         LAYER_PARSER(parseScale)},
        {"Deconvolution",                                      LAYER_PARSER(parseDeconvolution)},
        {"Power",                                              LAYER_PARSER(parsePower)},
        {"Copy",                                               LAYER_PARSER(parseCopy)},
        {"ELU",                                                LAYER_PARSER(parseELU)},
        // Flatten, Squeeze and Unsqueeze are represented as Reshape in VPU model
        {"Reshape",                                            LAYER_PARSER(parseReshape)},
        {"Flatten",                                            LAYER_PARSER(parseReshape)},
        {"Squeeze",                                            LAYER_PARSER(parseReshape)},
        {"Unsqueeze",                                          LAYER_PARSER(parseReshape)},
        {"Crop",                                               LAYER_PARSER(parseCrop)},
        {"Tile",                                               LAYER_PARSER(parseTile)},
        {"Normalize",                                          LAYER_PARSER(parseNormalize)},
        {"PriorBox",                                           LAYER_PARSER(parsePriorBox)},
        {"PriorBoxClustered",                                  LAYER_PARSER(parsePriorBoxClustered)},
        {"Permute",                                            LAYER_PARSER(parsePermute)},
        {"DetectionOutput",                                    LAYER_PARSER(parseDetectionOutput)},
        {"RegionYolo",                                         LAYER_PARSER(parseRegionYolo)},
        {"ReorgYolo",                                          LAYER_PARSER(parseReorgYolo)},
        {"CTCGreedyDecoder",                                   LAYER_PARSER(parseCTCDecoder)},
        {"Proposal",                                           LAYER_PARSER(parseProposal)},
        {"ROIPooling",                                         LAYER_PARSER(parseROIPooling)},
        {"PSROIPooling",                                       LAYER_PARSER(parsePSROIPooling)},
        {"Interp",                                             LAYER_PARSER(parseInterp)},
        {"Custom",                                             LAYER_PARSER(parseCustom)},
        {"MTCNN",                                              LAYER_PARSER(parseMTCNN)},
        {"LSTMCell",                                           LAYER_PARSER(parseLSTMCell)},
        {"Pad",                                                LAYER_PARSER(parsePad)},
        {"Resample",                                           LAYER_PARSER(parseResample)},
        {"LSTMSequence",                                       LAYER_PARSER(parseRNN)},
        {"GEMM",                                               LAYER_PARSER(parseGEMM)},
        {"Log",                                                LAYER_PARSER(parseLog)},
        {"Exp",                                                LAYER_PARSER(parseExp)},
        {"ReverseSequence",                                    LAYER_PARSER(parseReverseSequence)},
        {"Gather",                                             LAYER_PARSER(parseGather)},
        {"ReduceAnd",                                          LAYER_PARSER(parseReduce)},
        {"Floor",                                              LAYER_PARSER(parseFloor)},
        {"TopK",                                               LAYER_PARSER(parseTopK)},
        {"ReduceMin",                                          LAYER_PARSER(parseReduce)},
        {"StridedSlice",                                       LAYER_PARSER(parseStridedSlice)},
        {"Select",                                             LAYER_PARSER(parseSelect)},
        {"Erf",                                                LAYER_PARSER(parseErf)},
        {"ExperimentalDetectronDetectionOutput",               LAYER_PARSER(parseExpDetectionOutput)},
        {"NonMaxSuppression",                                  LAYER_PARSER(parseNonMaxSuppression)},
        {"ExperimentalDetectronROIFeatureExtractor",           LAYER_PARSER(parseROIFeatureExtractor)},
        {"Convert",                                            LAYER_PARSER(parseConvert)},
        {"ReduceMax",                                          LAYER_PARSER(parseReduce)},
        {"ReduceSum",                                          LAYER_PARSER(parseReduce)},
        {"ReduceMean",                                         LAYER_PARSER(parseReduce)},
        {"TensorIterator",                                     LAYER_PARSER(parseTensorIterator)},
        {"OneHot",                                             LAYER_PARSER(parseOneHot)},
        {"ExperimentalDetectronPriorGridGenerator",            LAYER_PARSER(parseExpPriorGridGenerator)},
        {"ExperimentalDetectronGenerateProposalsSingleImage",  LAYER_PARSER(parseExpGenerateProposals)},
    }} {}

ModelPtr FrontEnd::buildInitialModel(ie::ICNNNetwork& network) {
    VPU_PROFILE(buildInitialModel);

    const auto& env = CompileEnv::get();
    env.log->debug("FrontEnd : Build initial Model");
    VPU_LOGGER_SECTION(env.log);

    return runCommonPasses(network);
}

std::set<std::string> FrontEnd::checkSupportedLayers(ie::ICNNNetwork& network) {
    VPU_PROFILE(checkSupportedLayers);

    const auto& env = CompileEnv::get();

    env.log->debug("FrontEnd : Check supported layers");
    VPU_LOGGER_SECTION(env.log);

    std::set<std::string> supportedLayers;

    const auto onSupportedLayer = [&supportedLayers](const ie::CNNLayerPtr& layer) {
        supportedLayers.insert(layer->name);
    };

    const auto onUnsupportedLayer = [this](
        const Model& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs,
        const std::string& /*extraMsg*/) {
        _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
    };

    runCommonPasses(network, onUnsupportedLayer, onSupportedLayer);

    return supportedLayers;
}

namespace {

std::atomic<int> g_counter(0);

bool hasSuitableCustom(
        const std::vector<CustomLayer::Ptr>& customLayers,
        const ie::CNNLayerPtr& layer) {
    const auto& env = CompileEnv::get();
    ie::details::CaselessEq<std::string> cmp;

    env.log->trace("Check for suitable custom implementation for layer %s:%s", layer->name, layer->type);
    VPU_LOGGER_SECTION(env.log);

    for (const auto& customLayer : customLayers) {
        env.log->trace("Check next custom layer : %v", customLayer->whereParams());
        VPU_LOGGER_SECTION(env.log);

        bool suitable = true;
        for (const auto& whereParam : customLayer->whereParams()) {
            const auto iter = layer->params.find(whereParam.first);
            if (iter == layer->params.end() || !cmp(iter->second, whereParam.second)) {
                suitable = false;
                break;
            }
        }

        if (suitable) {
            env.log->trace("Matches");
            return true;
        }
    }

    return false;
}

}  // namespace

void FrontEnd::parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) {
    parseLayer(model, layer, inputs, outputs,
        [this](const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                            const std::string& extraMessage)
        { defaultOnUnsupportedLayerCallback(model, layer, inputs, outputs, extraMessage); });
}

void FrontEnd::parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                          const FrontEnd::UnsupportedLayerCallback& onUnsupported, const FrontEnd::SupportedLayerCallback& onSupported) {
    const auto customLayerByType  = _customLayers.find(layer->type);
    const auto customLayerAsStage = _customLayers.find(layer->type + "@stage_0");

    const bool isCustomLayer =
        ((customLayerByType != _customLayers.end()) && hasSuitableCustom(customLayerByType->second, layer)) ||
        ((customLayerAsStage != _customLayers.end()) && hasSuitableCustom(customLayerAsStage->second, layer));

    const auto& type = isCustomLayer ? "Custom" : layer->type;
    if (parsers.count(type) == 0) {
        if (onUnsupported) {
            onUnsupported(model, layer, inputs, outputs, formatString("unsupported layer type \"%v\"", type));
        }
        return;
    }

    try {
        parsers.at(type)(model, layer, inputs, outputs);
        if (onSupported) {
            onSupported(layer);
        }
    } catch (const details::UnsupportedLayerException&) {
        throw;
    } catch (const std::exception& error) {
        if (onUnsupported) {
            onUnsupported(model, layer, inputs, outputs, error.what());
        }
    }
}

void FrontEnd::defaultOnUnsupportedLayerCallback(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                                                 const std::string& extraMessage) {
    const auto& env = CompileEnv::get();
    VPU_THROW_UNSUPPORTED_UNLESS(env.config.ignoreUnknownLayers, "Failed to compile layer \"%v\": %v", layer->name, extraMessage);
    _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
}

ModelPtr FrontEnd::runCommonPasses(ie::ICNNNetwork& network) {
    return runCommonPasses(network, [this](const Model& model, const ie::CNNLayerPtr& layer,
                                                             const DataVector& inputs, const DataVector& outputs, const std::string& extraMessage)
        { defaultOnUnsupportedLayerCallback(model, layer, inputs, outputs, extraMessage); });
}


ModelPtr FrontEnd::runCommonPasses(ie::ICNNNetwork& network, const UnsupportedLayerCallback& unsupportedLayer, const SupportedLayerCallback& supportedLayer) {
    const auto& env = CompileEnv::get();

    //
    // Clear Front-end state
    //

    _ieParsedNetwork = {};
    _unbatchedOutputs.clear();
    _ieToVpuMap.clear();
    _customLayers.clear();
    _kernelNodes.clear();
    _lstmWeights.clear();
    _lstmBiases.clear();

    //
    // Parse custom layers
    //

    if (!env.config.customLayers.empty()) {
        env.log->trace("Parse custom layers : %s", env.config.customLayers);
        VPU_LOGGER_SECTION(env.log);

        if (env.platform != Platform::MYRIAD_X) {
            VPU_THROW_FORMAT("Custom layers are not supported for %v platforms", env.platform);
        }

        _customLayers = CustomLayer::loadFromFile(env.config.customLayers);
    }

    //
    // Create new VPU model
    //

    const auto model = std::make_shared<ModelObj>(network.getName());

    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    if (!env.config.ignoreIRStatistic) {
        ie::ICNNNetworkStats* stats = nullptr;
        // V10 IRs doesn't contain stats
        if (network.getStats(&stats, nullptr) == InferenceEngine::OK && !stats->isEmpty()) {
            env.log->trace("Use node statistics from the IR");
            model->setNodesStats(stats->getNodesStats());
        }
    }

    //
    // Update IE Network
    //

    {
        env.log->trace("Update IE Network");
        VPU_LOGGER_SECTION(env.log);

        IE_SUPPRESS_DEPRECATED_START
        // If we have NGraph network, but CNN compatibility is enabled, enforce conversion
        if (network.getFunction() && env.config.forceDeprecatedCnnConversion)
            network.addLayer(nullptr);
        IE_SUPPRESS_DEPRECATED_END

        detectNetworkBatch(network, model);

        ie::NetPass::ConvertPrecision(network, ie::Precision::I64, ie::Precision::I32);
        ie::NetPass::ConvertPrecision(network, ie::Precision::U64, ie::Precision::I32);
        ie::NetPass::ConvertPrecision(network, ie::Precision::BOOL, ie::Precision::I32);

        IE_SUPPRESS_DEPRECATED_START
        // force conversion to CNNNetwork
        if (network.getFunction())
            network.addLayer(nullptr);
        IE_SUPPRESS_DEPRECATED_END

        moveConstInputsToBlobs(network);

        removeConstLayers(network);

        unrollLoops(network);
    }

    //
    // Parse IR Network
    //

    _ieParsedNetwork = parseNetwork(network);

    //
    // Process internal VPU Model
    //

    {
        env.log->trace("Process internal VPU Model");
        VPU_LOGGER_SECTION(env.log);

        parseInputAndOutputData(model);

        addDataTypeConvertStages(model);

        addPreProcessStages(model);
    }

    //
    // Parse original layers
    //

    env.log->trace("Parse original layers");

    DataVector inputs, outputs;
    for (const auto& layer : origLayers()) {
        VPU_LOGGER_SECTION(env.log);

        env.log->trace("Try to parse layer %s:%s", layer->name, layer->type);
        VPU_LOGGER_SECTION(env.log);

        getInputAndOutputData(model, layer, inputs, outputs);

        if (env.config.skipAllLayers() || env.config.skipLayerType(layer->type)) {
            _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
            supportedLayer(layer);
            continue;
        }

        parseLayer(model, layer, inputs, outputs, unsupportedLayer, supportedLayer);
    }

    //
    // Clean up internal VPU Model
    //

    {
        env.log->trace("Clean up internal VPU Model");
        VPU_LOGGER_SECTION(env.log);

        model->cleanUp();
    }

    return model;
}

Data FrontEnd::getVpuData(const ie::DataPtr& ieData) const {
    IE_ASSERT(ieData != nullptr);

    const auto it = _ieToVpuMap.find(ieData);
    if (it == _ieToVpuMap.end()) {
        return nullptr;
    }

    return it->second;
}

void FrontEnd::bindData(const Data& data, const ie::DataPtr& ieData) {
    _ieToVpuMap[ieData] = data;
    data->setOrigData(ieData);
}

void FrontEnd::getInputAndOutputData(
        const Model& model,
        const ie::CNNLayerPtr& layer,
        DataVector& inputs,
        DataVector& outputs) {
    IE_ASSERT(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        const auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);

        inputs[i] = getVpuData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }

    outputs.resize(layer->outData.size());
    for (size_t i = 0; i < layer->outData.size(); ++i) {
        const auto layerOutput = layer->outData[i];
        IE_ASSERT(layerOutput != nullptr);

        if (const auto data = getVpuData(layerOutput)) {
            outputs[i] = data;
        } else {
            DataDesc dataDesc(layerOutput->getTensorDesc());
            if (dataDesc.type() == DataType::FP32) {
                // To infer the same FP32 models on different devices (CPU, GPU, VPU and so on)
                dataDesc.setType(DataType::FP16);
            }

            // Skip adding data if it not utilized
            const bool isNetworkOutput = _ieParsedNetwork.networkOutputs.count(layerOutput->getName()) > 0;
            const auto isLeaf = layerOutput->getInputTo().empty();
            if (!isNetworkOutput && isLeaf) {
                outputs[i] = nullptr;
                continue;
            }

            outputs[i] = model->addNewData(
                layerOutput->getName(),
                dataDesc);

            bindData(outputs[i], layerOutput);
        }
    }
}

std::tuple<Data, Data> FrontEnd::getWeightsAndBiases(const Model& model, const ie::CNNLayerPtr& layer) const {
    const auto baseLayer = std::dynamic_pointer_cast<ie::WeightableLayer>(layer);
    IE_ASSERT(baseLayer != nullptr);

    const auto origWeights = baseLayer->_weights;
    VPU_THROW_UNLESS(origWeights != nullptr, "Layer %s has no weights", layer->name);

    const auto weights = model->addConstData(
        layer->name + "@weights",
        DataDesc({origWeights->size()}),
        ieBlobContent(origWeights));

    const auto origBiases = baseLayer->_biases;

    Data biases;
    if (origBiases == nullptr) {
        biases = model->addFakeData();
    } else {
        biases = model->addConstData(
            layer->name + "@biases",
            DataDesc({origBiases->size()}),
            ieBlobContent(origBiases));
    }

    return std::make_tuple(weights, biases);
}

}  // namespace vpu
