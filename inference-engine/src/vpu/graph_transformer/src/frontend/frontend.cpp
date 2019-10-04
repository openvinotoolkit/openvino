// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <map>
#include <vector>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

typedef void (FrontEnd::*parser_t)(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        const DataVector& inputs,
        const DataVector& outputs);

ie::details::caseless_map<std::string, parser_t> g_parsers = {
    {"Convolution",                              &FrontEnd::parseConvolution},
    {"Pooling",                                  &FrontEnd::parsePooling},
    {"ReLU",                                     &FrontEnd::parseReLU},
    {"Clamp",                                    &FrontEnd::parseClamp},
    {"FullyConnected",                           &FrontEnd::parseFullyConnected},
    {"SoftMax",                                  &FrontEnd::parseSoftMax},
    {"GRN",                                      &FrontEnd::parseGRN},
    {"MVN",                                      &FrontEnd::parseMVN},
    {"Norm",                                     &FrontEnd::parseNorm},
    {"Concat",                                   &FrontEnd::parseConcat},
    {"Eltwise",                                  &FrontEnd::parseEltwise},
    {"Split",                                    &FrontEnd::parseSplit},
    {"Sigmoid",                                  &FrontEnd::parseSigmoid},
    {"TanH",                                     &FrontEnd::parseTanH},
    {"PReLU",                                    &FrontEnd::parsePReLU},
    {"Bias",                                     &FrontEnd::parseBias},
    {"Slice",                                    &FrontEnd::parseSplit},  // Caffe Slice is transformed to Split by IE
    {"BatchNormalization",                       &FrontEnd::parseBatchNorm},
    {"ScaleShift",                               &FrontEnd::parseScale},
    {"Deconvolution",                            &FrontEnd::parseDeconvolution},
    {"Power",                                    &FrontEnd::parsePower},
    {"Copy",                                     &FrontEnd::parseCopy},
    {"ELU",                                      &FrontEnd::parseELU},

    // Flatten, Squeeze and Unsqueeze are represented as Reshape in VPU model
    {"Reshape",                                  &FrontEnd::parseReshape},
    {"Flatten",                                  &FrontEnd::parseReshape},
    {"Squeeze",                                  &FrontEnd::parseReshape},
    {"Unsqueeze",                                &FrontEnd::parseReshape},

    {"Crop",                                     &FrontEnd::parseCrop},
    {"Tile",                                     &FrontEnd::parseTile},
    {"Normalize",                                &FrontEnd::parseNormalize},
    {"PriorBox",                                 &FrontEnd::parsePriorBox},
    {"PriorBoxClustered",                        &FrontEnd::parsePriorBoxClustered},
    {"Permute",                                  &FrontEnd::parsePermute},
    {"DetectionOutput",                          &FrontEnd::parseDetectionOutput},
    {"RegionYolo",                               &FrontEnd::parseRegionYolo},
    {"ReorgYolo",                                &FrontEnd::parseReorgYolo},
    {"CTCGreedyDecoder",                         &FrontEnd::parseCTCDecoder},
    {"Proposal",                                 &FrontEnd::parseProposal},
    {"ROIPooling",                               &FrontEnd::parseROIPooling},
    {"PSROIPooling",                             &FrontEnd::parsePSROIPooling},
    {"Interp",                                   &FrontEnd::parseInterp},
    {"Custom",                                   &FrontEnd::parseCustom},
    {"MTCNN",                                    &FrontEnd::parseMTCNN},
    {"LSTMCell",                                 &FrontEnd::parseLSTMCell},
    {"Pad",                                      &FrontEnd::parsePad},
    {"Resample",                                 &FrontEnd::parseResample},
    {"ArgMax",                                   &FrontEnd::parseArgMax},
    {"LSTMSequence",                             &FrontEnd::parseRNN},
    {"GEMM",                                     &FrontEnd::parseGEMM},
    {"Log",                                      &FrontEnd::parseLog},
    {"Exp",                                      &FrontEnd::parseExp},
    {"ReverseSequence",                          &FrontEnd::parseReverseSequence},
    {"Gather",                                   &FrontEnd::parseGather},
    {"ReduceAnd",                                &FrontEnd::parseReduce},
    {"Floor",                                    &FrontEnd::parseFloor},
    {"TopK",                                     &FrontEnd::parseTopK},
    {"ReduceMin",                                &FrontEnd::parseReduce},
    {"StridedSlice",                             &FrontEnd::parseStridedSlice},
    {"Select",                                   &FrontEnd::parseSelect},
    {"ExperimentalDetectronDetectionOutput",     &FrontEnd::parseExpDetectionOutput},
    {"NonMaxSuppression",                        &FrontEnd::parseNonMaxSuppression},
    {"ExperimentalDetectronROIFeatureExtractor", &FrontEnd::parseROIFeatureExtractor},
};

std::atomic<int> g_counter(0);

}  // namespace

void FrontEnd::eliminatePriorBoxData(const Model::Ptr& model) {
    VPU_PROFILE(eliminatePriorBoxData);

    auto isConvertStage = [](StageType stage) {
        return stage == StageType::Convert_u8f16  ||
               stage == StageType::Convert_f32f16 ||
               stage == StageType::Convert_f16f32;
    };

    auto isPriorBox = [](std::string type) {
        return ie::details::CaselessEq<std::string>()(type, "PriorBox") ||
               ie::details::CaselessEq<std::string>()(type, "PriorBoxClustered");
    };

    for (const auto& data : model->datas()) {
        if (data->usage() == DataUsage::Input) {
            auto consumers_num = data->numConsumers();
            bool unused = (0 == consumers_num);

            // If data has consumer it still could be just data conversion stage
            if (consumers_num == 1) {
                auto stage = data->singleConsumer();
                if (isConvertStage(stage->type())) {
                    IE_ASSERT(stage->numOutputs() == 1);

                    auto output = stage->output(0);
                    if (output->numConsumers() == 0) {
                        unused = true;
                    }
                }
            }

            if (unused) {
                auto origData = data->origData();
                IE_ASSERT(origData != nullptr);
                IE_ASSERT(!origData->getInputTo().empty());

                bool priorBox = true;
                for (const auto& consumer_it : origData->getInputTo()) {
                    auto consumer = consumer_it.second;
                    priorBox &= isPriorBox(consumer->type);
                }

                if (priorBox) {
                    if (1 == consumers_num) {
                        model->removeStage(data->singleConsumer());
                    }
                    model->removeUnusedData(data);
                }
            }
        }
    }
}

static bool hasSuitableCustom(const std::vector<CustomLayer::Ptr>& customLayers,
                              const std::map<std::string, std::string>& layerParams) {
    ie::details::CaselessEq<std::string> cmp;

    for (const auto & customLayer : customLayers) {
        bool suitable = true;
        for (auto whereParam : customLayer->whereParams()) {
            if (layerParams.find(whereParam.first) == layerParams.end() || !cmp(layerParams.find(whereParam.first)->second, whereParam.second)) {
                suitable = false;
            }
        }
        if (suitable) {
            return true;
        }
    }

    return false;
}

Model::Ptr FrontEnd::buildInitialModel(const ie::ICNNNetwork& network) {
    const auto& env = CompileEnv::get();

    env.log->debug("Build initial Model");
    VPU_LOGGER_SECTION(env.log);

    auto model = runCommonPasses(network, LayersOrder::DFS);

    DataVector inputs, outputs;
    for (const auto& layer : _ieNetworkParser.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        env.log->debug("Try to parse layer [%s]", layer->name);

        getInputAndOutputData(model, layer, inputs, outputs);

        if (env.netConfig.skipAllLayers() ||
            env.netConfig.skipLayerType(layer->type)) {
            _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
            continue;
        }

        auto customLayerFound0 = _customLayers.find(layer->type);
        auto customLayerFound1 = _customLayers.find(layer->type + "@stage_0");

        auto it = (((customLayerFound0 != _customLayers.end()) && hasSuitableCustom(customLayerFound0->second, layer->params)) ||
                   ((customLayerFound1 != _customLayers.end()) && hasSuitableCustom(customLayerFound1->second, layer->params)) )
                   ? g_parsers.find("Custom")
                   : g_parsers.find(layer->type);

        if (it == g_parsers.end()) {
            if (env.config.ignoreUnknownLayers) {
                _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
                continue;
            } else {
                VPU_THROW_EXCEPTION
                        << "Cannot convert layer \""
                        << layer->name
                        << "\" due to unsupported layer type \""
                        << layer->type
                        << "\"";
            }
        }

        auto parser = it->second;
        IE_ASSERT(parser != nullptr);

        (this->*parser)(model, layer, inputs, outputs);
    }

    eliminatePriorBoxData(model);

    model->cleanUp();

    return model;
}

std::set<std::string> FrontEnd::checkSupportedLayers(const ie::ICNNNetwork& network) {
    const auto& env = CompileEnv::get();

    auto model = runCommonPasses(network, LayersOrder::BFS);

    std::set<std::string> layerNames;

    DataVector inputs, outputs;
    for (const auto& layer : _ieNetworkParser.orderedLayers) {
        IE_ASSERT(layer != nullptr);

        env.log->debug("Try to parse layer %s", layer->name);

        getInputAndOutputData(model, layer, inputs, outputs);

        auto it =
                (_customLayers.count(layer->type) > 0) ?
                    g_parsers.find("Custom") :
                    g_parsers.find(layer->type);
        if (it != g_parsers.end()) {
            try {
                // If we can create and have not thrown exception, then layer is supported.
                auto parser = it->second;
                IE_ASSERT(parser != nullptr);

                (this->*parser)(model, layer, inputs, outputs);

                layerNames.insert(layer->name);
            } catch (const ie::details::InferenceEngineException&) {
                // Nothing to do
                continue;
            }
        }
    }

    return layerNames;
}

Model::Ptr FrontEnd::runCommonPasses(
        const ie::ICNNNetwork& network,
        LayersOrder order) {
    const auto& env = CompileEnv::get();

    //
    // Load Custom layers
    //

    if (!env.config.customLayers.empty()) {
        if (env.platform == Platform::MYRIAD_2) {
            VPU_LOG_AND_THROW(env.log, "Custom layers are not supported for Myriad 2 platforms");
        }

        _customLayers = CustomLayer::loadFromFile(env.config.customLayers);
    }

    //
    // Clear Front-end state
    //

    _ieNetworkParser.clear();
    _unbatchedOutputs.clear();
    _ieToVpuMap.clear();

    //
    // Create new VPU model
    //

    auto model = std::make_shared<Model>(network.getName());

    if (!env.config.ignoreIRStatistic) {
        InferenceEngine::ICNNNetworkStats* stats = nullptr;
        if (InferenceEngine::StatusCode::OK == network.getStats(&stats, nullptr) && !stats->isEmpty()) {
            model->setNodesStats(stats->getNodesStats());
        }
    }

    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    //
    // Detect network batch
    //

    auto reshapedNetwork = detectNetworkBatch(network, model);

    //
    // Remove constant layers from network
    //

    RemoveConstLayers(reshapedNetwork);

    //
    // Get IE layers in topological order
    //

    if (order == LayersOrder::DFS) {
        _ieNetworkParser.parseNetworkDFS(reshapedNetwork);
    } else {
        _ieNetworkParser.parseNetworkBFS(reshapedNetwork);
    }

    //
    // Parse network inputs/outputs/const datas
    //

    parseInputAndOutputData(model);

    //
    // Add data type convert stages
    //

    addDataTypeConvertStages(model);

    //
    // Add pre-process stages
    //

    addPreProcessStages(model);

    return model;
}

Data FrontEnd::getVpuData(const ie::DataPtr& ieData) {
    IE_ASSERT(ieData != nullptr);

    auto it = _ieToVpuMap.find(ieData);
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
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer,
        DataVector& inputs,
        DataVector& outputs) {
    IE_ASSERT(layer != nullptr);

    inputs.resize(layer->insData.size());
    for (size_t i = 0; i < layer->insData.size(); ++i) {
        auto layerInput = layer->insData[i].lock();
        IE_ASSERT(layerInput != nullptr);

        inputs[i] = getVpuData(layerInput);
        IE_ASSERT(inputs[i] != nullptr);
    }

    outputs.resize(layer->outData.size());
    for (size_t i = 0; i < layer->outData.size(); ++i) {
        auto layerOutput = layer->outData[i];
        IE_ASSERT(layerOutput != nullptr);

        if (auto data = getVpuData(layerOutput)) {
            outputs[i] = data;
        } else {
            DataDesc dataDesc(layerOutput->getTensorDesc());
            if (dataDesc.type() == DataType::FP32) {
                // To infer the same FP32 models on different devices (CPU, GPU, VPU and so on)
                dataDesc.setType(DataType::FP16);
            }

            outputs[i] = model->addNewData(
                layerOutput->getName(),
                dataDesc);

            bindData(outputs[i], layerOutput);
        }
    }
}

std::tuple<Data, Data> FrontEnd::getWeightsAndBiases(
        const Model::Ptr& model,
        const ie::CNNLayerPtr& layer) {
    auto baseLayer = std::dynamic_pointer_cast<ie::WeightableLayer>(layer);
    IE_ASSERT(baseLayer != nullptr);

    auto origWeights = baseLayer->_weights;
    if (origWeights == nullptr) {
        THROW_IE_EXCEPTION << "weights are empty for layer: " << layer->name;
    }

    auto weights = model->addConstData(
        layer->name + "@weights",
        DataDesc({origWeights->size()}),
        ieBlobContent(origWeights));

    auto origBiases = baseLayer->_biases;

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
