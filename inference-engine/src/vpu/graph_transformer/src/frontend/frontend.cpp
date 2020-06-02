// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/utils/profiling.hpp"
#include "vpu/compile_env.hpp"
#include "vpu/model/data_contents/ie_blob_content.hpp"

#include "net_pass.h"

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include <string>

#include <convert_function_to_cnn_network.hpp>
#include <generic_ie.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>

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
        {"ScatterUpdate",                                      LAYER_PARSER(parseScatterUpdate)},
        {"ScatterElementsUpdate",                              LAYER_PARSER(parseScatterElementsUpdate)},
        {"ExperimentalDetectronTopKROIs",                      LAYER_PARSER(parseExpTopKROIs)},
        {"StaticShapeNonZero",                                 LAYER_PARSER(parseNonZero)},
        {"ROIAlign",                                           LAYER_PARSER(parseROIAlign)},
        {"DynamicShapeResolver",                               LAYER_PARSER(parseDSR)},
        {"OutShapeOfReshape",                                  LAYER_PARSER(parseOutShapeOfReshape)},
        {"StaticShapeBroadcast",                               LAYER_PARSER(parseBroadcast)},
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

}  // namespace

CustomLayer::Ptr FrontEnd::getSuitableCustomLayer(const std::vector<CustomLayer::Ptr>& customLayers,
                                                  const ie::CNNLayerPtr& cnnLayer) {
    const auto& env = CompileEnv::get();
    env.log->trace("Check for suitable custom implementation for layer %s:%s",
                   cnnLayer->name, cnnLayer->type);
    VPU_LOGGER_SECTION(env.log);

    const auto cnnInputs = [&] {
        auto inputs = SmallVector<CustomDataFormat>{};
        inputs.reserve(cnnLayer->insData.size());
        for (const auto& input : cnnLayer->insData) {
            const auto layout = input.lock()->getLayout();
            const auto format = CustomLayer::formatFromLayout(layout);
            inputs.push_back(format);
        }
        return inputs;
    }();

    const auto cnnOutputs = [&] {
        auto outputs = SmallVector<CustomDataFormat>{};
        outputs.reserve(cnnLayer->outData.size());
        for (const auto& output : cnnLayer->outData) {
            const auto layout = output->getLayout();
            const auto format = CustomLayer::formatFromLayout(layout);
            outputs.push_back(format);
        }
        return outputs;
    }();

    const auto isSuitableLayer = [&env, &cnnLayer](const CustomLayer::Ptr& customLayer) {
        env.log->trace("Check next custom layer : %v", customLayer->layerName());
        VPU_LOGGER_SECTION(env.log);

        if (!customLayer->meetsWhereRestrictions(cnnLayer->params)) {
            env.log->trace("Where restrictions are not met");
            return false;
        }

        for (const auto& kernel : customLayer->kernels()) {
            const auto& gws = kernel.globalGridSizeRules();
            const auto& lws = kernel.localGridSizeRules();

            const auto validSizeRule = [&](const std::string& rule) {
                return CustomLayer::isLegalSizeRule(rule, cnnLayer->params);
            };

            const auto validGridSizes = std::all_of(begin(gws), end(gws), validSizeRule) &&
                                        std::all_of(begin(lws), end(lws), validSizeRule);

            if (!validGridSizes) {
                env.log->trace("Work group grid sizes are not valid");
                return false;
            }
        }

        return true;
    };

    auto suitableCustomLayers = SmallVector<CustomLayer::Ptr>{};

    std::copy_if(begin(customLayers), end(customLayers),
        back_inserter(suitableCustomLayers), isSuitableLayer);

    if (suitableCustomLayers.empty()) {
      return nullptr;
    }

    const auto inputsLayoutMatch = [&](const SmallVector<CustomDataFormat>& cnnEdges,
                                       const std::map<int, CustomDataFormat>& clEdges) {
        for (const auto clEdge : clEdges) {
            const auto port = clEdge.first;
            VPU_THROW_UNLESS(port < cnnEdges.size(),
                "Can't bind custom layer edge with port '%s' to CNNNetwork layer", port);

            const auto clFormat = clEdge.second;
            const auto cnnFormat = cnnEdges[port];
            if (cnnFormat != clFormat &&
                cnnFormat != CustomDataFormat::Any &&
                clFormat != CustomDataFormat::Any) {
                return false;
            }
        }
        return true;
    };


    for (const auto& customLayer : suitableCustomLayers) {
        const auto clInputs = customLayer->inputs();

        if (inputsLayoutMatch(cnnInputs, clInputs)) {
            env.log->trace("Found suitable '%s' custom layer", customLayer->layerName());
            return customLayer;
        }
    }

    const auto firstGoodLayer = suitableCustomLayers.front();
    env.log->trace("Found suitable custom layer '%s', but input layouts "
                   "have not matched with what CNNNetwork expected",
                   firstGoodLayer->layerName());
    return firstGoodLayer;
}


void FrontEnd::parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) {
    parseLayer(model, layer, inputs, outputs,
        [this](const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                            const std::string& extraMessage)
        { defaultOnUnsupportedLayerCallback(model, layer, inputs, outputs, extraMessage); });
}

void FrontEnd::parseLayer(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                          const FrontEnd::UnsupportedLayerCallback& onUnsupported, const FrontEnd::SupportedLayerCallback& onSupported) {
    const auto customLayer = _customLayers.find(layer->type);
    const bool isCustomLayer = customLayer != _customLayers.end() && getSuitableCustomLayer(customLayer->second, layer);

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
    // NGraph -> CNN conversion may be called in 2 different moments: at
    // the beginning if conversion was forced by configuration or after detect
    // network batch and precision conversions. Conversion utility
    // returns std::shared_ptr. ICNNNetwork is neither copyable nor movable.
    // As a result, it is impossible to overwrite given "network" argument.
    // Do not use network parameter in this function to avoid using wrong network
    // reference (e.g. original instead of converted).
    auto* originalOrConvertNetwork = &network;

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

    const auto model = std::make_shared<ModelObj>(originalOrConvertNetwork->getName());

    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    if (!env.config.ignoreIRStatistic) {
        ie::ICNNNetworkStats* stats = nullptr;
        // V10 IRs doesn't contain stats
        if (originalOrConvertNetwork->getStats(&stats, nullptr) == InferenceEngine::OK && !stats->isEmpty()) {
            env.log->trace("Use node statistics from the IR");
            model->setNodesStats(stats->getNodesStats());
        }
    }

    //
    // Update IE Network
    //

    std::shared_ptr<ie::ICNNNetwork> convertedNetwork;

    {
        env.log->trace("Update IE Network");
        VPU_LOGGER_SECTION(env.log);

        auto convertNetwork = [&convertedNetwork, &originalOrConvertNetwork]() {
            auto nGraphFunc = originalOrConvertNetwork->getFunction();
            // Disable shape inference (WA for generic operations)
            ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

            ngraph::pass::CommonOptimizations().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet3ToOpSet2().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet2ToOpSet1().run_on_function(nGraphFunc);
            ngraph::pass::ConvertOpSet1ToLegacy().run_on_function(nGraphFunc);
            convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *originalOrConvertNetwork);
            originalOrConvertNetwork = convertedNetwork.get();
        };

        if (originalOrConvertNetwork->getFunction() && env.config.forceDeprecatedCnnConversion) {
            convertNetwork();
        }

        detectNetworkBatch(*originalOrConvertNetwork, model);

        if (originalOrConvertNetwork->getFunction()) {
            convertNetwork();
        }

        ie::NetPass::ConvertPrecision(*originalOrConvertNetwork, ie::Precision::I64, ie::Precision::I32);
        ie::NetPass::ConvertPrecision(*originalOrConvertNetwork, ie::Precision::U64, ie::Precision::I32);
        ie::NetPass::ConvertPrecision(*originalOrConvertNetwork, ie::Precision::BOOL, ie::Precision::I32);

        moveConstInputsToBlobs(*originalOrConvertNetwork);

        removeConstLayers(*originalOrConvertNetwork);

        unrollLoops(*originalOrConvertNetwork);
    }

    //
    // Parse IR Network
    //

    _ieParsedNetwork = parseNetwork(*originalOrConvertNetwork);

    //
    // Process internal VPU Model
    //

    {
        env.log->trace("Process internal VPU Model");
        VPU_LOGGER_SECTION(env.log);

        parseInputAndOutputData(model);

        if (!CompileEnv::get().config.disableConvertStages) {
            addDataTypeConvertStages(model);
        }

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
