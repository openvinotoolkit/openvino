// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/utils/profiling.hpp"
#include "vpu/compile_env.hpp"
#include "vpu/model/data_contents/ie_blob_content.hpp"

#include <legacy/net_pass.h>

#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <set>
#include <map>
#include <vector>
#include <utility>
#include <string>

#include <legacy/convert_function_to_cnn_network.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/op_conversions/convert_gelu.hpp>
#include <transformations/op_conversions/softplus_decomposition.hpp>
#include <transformations/op_conversions/convert_minimum_to_power_and_max.hpp>
#include <transformations/op_conversions/hswish_decomposition.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp>
#include <transformations/convert_precision.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/init_node_info.hpp>
#include <vpu/ngraph/transformations/convert_extract_image_patches_to_reorg_yolo.hpp>
#include <vpu/ngraph/transformations/merge_subsequent_dsr_operations.hpp>
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/transformations/eliminate_shapeof_after_dsr.hpp"
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/utilities.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <vpu/ngraph/transformations/extract_dynamic_batch/extract_dynamic_batch.hpp>
#include <vpu/ngraph/transformations/merge_gather_gather_elements.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
namespace vpu {
FrontEnd::FrontEnd(StageBuilder::Ptr stageBuilder, const ie::ICore* core)
    : _stageBuilder(std::move(stageBuilder)),
    _core(core),
    parsers{{
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
        {"Interpolate",                                        LAYER_PARSER(parseInterpolate)},
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
        {"StaticShapeNonMaxSuppression",                       LAYER_PARSER(parseStaticShapeNMS)},
        {"StaticShapeReshape",                                 LAYER_PARSER(parseReshape)},
        {"Mish",                                               LAYER_PARSER(parseMish)},
        {"Gelu",                                               LAYER_PARSER(parseGelu)},
        {"SoftPlus",                                           LAYER_PARSER(parseSoftPlus)},
        {"Swish",                                              LAYER_PARSER(parseSwish)},
        {"Activation",                                         LAYER_PARSER(parseActivation)},
        {"GatherND",                                           LAYER_PARSER(parseGatherND)},
        {"HSwish",                                             LAYER_PARSER(parseHSwish)},
        {"Ceiling",                                            LAYER_PARSER(parseCeiling)},
        {"GatherElements",                                     LAYER_PARSER(parseGatherElements)},
        {"ExpGatherElements",                                  LAYER_PARSER(parseGatherElements)},
        {"Round",                                              LAYER_PARSER(parseRound)},
        {"CTCGreedyDecoderSeqLen",                             LAYER_PARSER(parseCTCGreedyDecoderSeqLen)},
    }} {
        VPU_THROW_UNLESS(_core != nullptr, "Argument core is null");
    }

ModelPtr FrontEnd::buildInitialModel(const ie::CNNNetwork& network) {
    VPU_PROFILE(buildInitialModel);

    const auto& env = CompileEnv::get();
    env.log->debug("FrontEnd : Build initial Model");
    VPU_LOGGER_SECTION(env.log);

    return runCommonPasses(network);
}

ie::CNNNetwork FrontEnd::convertNetwork(ie::CNNNetwork& network) {
    auto nGraphFunc = network.getFunction();

    ngraph::pass::Manager manager;
    manager.register_pass<::ngraph::pass::InitNodeInfo>();
    // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
    manager.register_pass<::ngraph::pass::ConvertPriorBox>();
    manager.register_pass<ngraph::pass::ConvertNMS1ToNMS5>();
    manager.register_pass<ngraph::pass::ConvertNMS3ToNMS5>();
    manager.register_pass<ngraph::pass::ConvertNMS4ToNMS5>();
    manager.register_pass<vpu::MergeGatherGatherElements>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();

    manager.register_pass<vpu::ExtractBatch>(std::unordered_set<ngraph::Node::type_info_t> {
        ngraph::opset5::MatMul::type_info,
        ngraph::opset5::Convolution::type_info,
        ngraph::opset5::GroupConvolution::type_info
    });
    manager.register_pass<vpu::DynamicToStaticShape>();
    manager.register_pass<vpu::EliminateShapeOfAfterDSR>();
    manager.register_pass<vpu::ConvertExtractImagePatchesToReorgYolo>();
    // ConstantFolding placed here to avoid precision type missmatch when we try to evaluate nodes with BOOL output.
    // For example evaluate_greater_equal calls set_broadcast function with hardcoded BOOL precision.
    // In set_broadcast function we compare original node's precision with hardcoded so we get an error if we change precision before.
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    // ConvertPrecision must be executed before ConvertOpSet1ToLegacy due to this pass works with operations from opsets only
    static const precisions_array precisions = {
        { ngraph::element::i64, ngraph::element::i32 },
        { ngraph::element::u64, ngraph::element::i32 },
        { ngraph::element::u32, ngraph::element::i32 },
        { ngraph::element::boolean, ngraph::element::i32 }
    };
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions, myriadTypeToFuseMap);

    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    //  ConvertOpSet1ToLegacy can produce constants with I64 precision
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::i64, ngraph::element::i32 }}, myriadTypeToFuseMap);
    manager.register_pass<vpu::MergeSubsequentDSROperations>();

    auto pass_config = manager.get_pass_config();
    pass_config->disable<ngraph::pass::ConvertGatherToGatherIEMatcher>();
    pass_config->disable<ngraph::pass::ConvertGELU>();
    pass_config->disable<ngraph::pass::SoftPlusDecomposition>();
    pass_config->disable<ngraph::pass::ConvertMinimum>();
    pass_config->disable<ngraph::pass::HSwishDecomposition>();
    pass_config->disable<ngraph::pass::MVN6Decomposition>();
    pass_config->disable<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();

    auto transformationPredicate = [](const std::shared_ptr<const ngraph::Node>& node) -> bool {
        return !!std::dynamic_pointer_cast<const ngraph::vpu::op::DynamicShapeResolver>(node->input_value(0).get_node_shared_ptr());
    };
    pass_config->set_callback<ngraph::pass::ConvertMatMulToFC,
                              ngraph::pass::ConvertStridedSliceToCropMatcher>(transformationPredicate);

    manager.run_passes(nGraphFunc);
    IE_SUPPRESS_DEPRECATED_START
    return ie::CNNNetwork(ie::details::convertFunctionToICNNNetwork(nGraphFunc, network));
    IE_SUPPRESS_DEPRECATED_END
}

std::set<std::string> FrontEnd::checkSupportedLayers(const ie::CNNNetwork& network) {
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

    runCommonPasses(cloneNetwork(network), onUnsupportedLayer, onSupportedLayer);

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

void FrontEnd::processTrivialCases(const Model& model) {
    std::unordered_map<ie::DataPtr, std::pair<Data, Data>> ieDataToTrivialCase;
    for (const auto& data : model->datas()) {
        const auto& origData = data->origData();
        if (origData == nullptr) {
            continue;
        }

        auto& trivialCase = ieDataToTrivialCase[origData];
        auto& destination = data->usage() == DataUsage::Output ? trivialCase.second : trivialCase.first;
        VPU_THROW_UNLESS(ieDataToTrivialCase.count(origData) == 0 || destination == nullptr,
            "Encountered IE data object {} which has two vpu data objects {} and {} of the same type {} associated with it, while only one is permitted",
            origData->getName(), destination->name(), data->name(), destination->usage());
        destination = data;
    }

    for (const auto& trivialCase : ieDataToTrivialCase) {
        const auto& trivialCasePair = trivialCase.second;

        const auto& unconnectedInput = trivialCasePair.first;
        const auto& unconnectedOutput = trivialCasePair.second;

        if (!unconnectedInput || !unconnectedOutput) {
            continue;
        }

        _stageBuilder->addCopyStage(
            model,
            unconnectedInput->name() + "@copy",
            nullptr,
            {unconnectedInput},
            {unconnectedOutput},
            "processTrivialCase");
    }
}

void FrontEnd::defaultOnUnsupportedLayerCallback(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs,
                                                 const std::string& extraMessage) {
    const auto& env = CompileEnv::get();
    VPU_THROW_UNSUPPORTED_UNLESS(env.config.ignoreUnknownLayers, "Failed to compile layer \"%v\": %v", layer->name, extraMessage);
    _stageBuilder->addNoneStage(model, layer->name, layer, inputs, outputs);
}

ModelPtr FrontEnd::runCommonPasses(const ie::CNNNetwork& network) {
    return runCommonPasses(cloneNetwork(network),
        [this](const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs, const std::string& extraMessage) {
            defaultOnUnsupportedLayerCallback(model, layer, inputs, outputs, extraMessage);});
}

ModelPtr FrontEnd::runCommonPasses(ie::CNNNetwork network,
    const UnsupportedLayerCallback& unsupportedLayer, const SupportedLayerCallback& supportedLayer) {
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

    auto model = std::make_shared<ModelObj>(network.getName());

    model->attrs().set<int>("index", g_counter.fetch_add(1));
    model->attrs().set<Resources>("resources", env.resources);

    //
    // Update IE Network
    //

    {
        env.log->trace("Update IE Network");
        VPU_LOGGER_SECTION(env.log);

        if (network.getFunction() && env.config.forceDeprecatedCnnConversion) {
            network = convertNetwork(network);
        }

        detectNetworkBatch(network, model);

        if (network.getFunction()) {
            network = convertNetwork(network);
        }

        const std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> convert_precision_list {
            {ngraph::element::i64, ngraph::element::i32},
            {ngraph::element::u64, ngraph::element::i32},
            {ngraph::element::u32, ngraph::element::i32},
            {ngraph::element::boolean, ngraph::element::i32},
        };
        // WA: after conversion to CNNNetwork user precision can redefine input/output precisions
        // so we need to apply additional precision conversion but only for inputs and outputs
        // This method should be removed #-48878
        for (const auto& precision : convert_precision_list) {
            ie::NetPass::ConvertIOPrecision(network,
                                            InferenceEngine::details::convertPrecision(precision.first),
                                            InferenceEngine::details::convertPrecision(precision.second));
        }
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

        //
        // Process trivial cases like `input->output`, `const->output`
        //

        processTrivialCases(model);

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
            const auto isLeaf = getInputTo(layerOutput).empty();
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
