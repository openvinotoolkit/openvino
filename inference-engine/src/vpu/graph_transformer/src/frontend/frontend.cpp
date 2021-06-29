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
#include "vpu/ngraph/transformations/convert_matmul_to_fc.hpp"
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/utilities.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <vpu/ngraph/transformations/extract_dynamic_batch/extract_dynamic_batch.hpp>
#include <vpu/ngraph/transformations/merge_gather_gather_elements.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
/// debug
#include <ngraph/pass/visualize_tree.hpp>
namespace vpu {
FrontEnd::FrontEnd(StageBuilder::Ptr stageBuilder, const ie::ICore* core)
    : _stageBuilder(std::move(stageBuilder)),
    _core(core),
    parsers{{
        {"Convolution",                                        LAYER_PARSER(parseConvolution)},
        {"GroupConvolution",                                        LAYER_PARSER(parseGroupConvolution)},
        {"AvgPool",                                            LAYER_PARSER(parseAvgPooling)},
        {"MaxPool",                                            LAYER_PARSER(parseMaxPooling)},
        {"ReLU",                                               LAYER_PARSER(parseReLU)},
        {"Clamp",                                              LAYER_PARSER(parseClamp)},
        {"FullyConnected",                                     LAYER_PARSER(parseFullyConnected)},
        {"SoftMax",                                            LAYER_PARSER(parseSoftMax)},
        {"GRN",                                                LAYER_PARSER(parseGRN)},
        {"MVN",                                                LAYER_PARSER(parseMVN)},
        {"LRN",                                                LAYER_PARSER(parseNorm)},
        {"Concat",                                             LAYER_PARSER(parseConcat)},
        // {"Eltwise",                                            LAYER_PARSER(parseEltwise)},
        {"Subtract",                                           LAYER_PARSER(parseSubtract)},
        {"Add",                                                LAYER_PARSER(parseAdd)},
        {"Multiply",                                           LAYER_PARSER(parseMultiply)},
        {"Maximum",                                            LAYER_PARSER(parseMaximum)},
        {"Divide",                                             LAYER_PARSER(parseDivide)},
        {"Minimum",                                            LAYER_PARSER(parseMinimum)},
        {"SquaredDifference",                                  LAYER_PARSER(parseSquaredDifference)},
        {"Equal",                                              LAYER_PARSER(parseEqual)},
        {"NotEqual",                                           LAYER_PARSER(parseNotEqual)},
        {"Greater",                                            LAYER_PARSER(parseGreater)},
        {"GreaterEqual",                                       LAYER_PARSER(parseGreaterEqual)},
        {"Less",                                               LAYER_PARSER(parseLess)},
        {"LessEqual",                                          LAYER_PARSER(parseLessEqual)},
        {"LogicalNot",                                         LAYER_PARSER(parseLogicalNot)},
        {"LogicalAnd",                                         LAYER_PARSER(parseLogicalAnd)},
        {"LogicalOr",                                          LAYER_PARSER(parseLogicalOr)},
        {"LogicalXor",                                         LAYER_PARSER(parseLogicalXor)},
        // Slice is represented as Split in VPU model
        {"Split",                                              LAYER_PARSER(parseSplit)},
        {"Slice",                                              LAYER_PARSER(parseSplit)},
        {"Sigmoid",                                            LAYER_PARSER(parseSigmoid)},
        {"TanH",                                               LAYER_PARSER(parseTanH)},
        {"PReLU",                                              LAYER_PARSER(parsePReLU)},
        {"Bias",                                               LAYER_PARSER(parseBias)},
        // {"BatchNormalization",                                 LAYER_PARSER(parseBatchNorm)},
        // {"ScaleShift",                                         LAYER_PARSER(parseScale)},
        {"Deconvolution",                                      LAYER_PARSER(parseDeconvolution)},
        {"Power",                                              LAYER_PARSER(parsePower)},
        {"Sqrt",                                               LAYER_PARSER(parseSqrt)},
        {"Copy",                                               LAYER_PARSER(parseCopy)},
        {"ELU",                                                LAYER_PARSER(parseELU)},
        // Flatten, Squeeze and Unsqueeze are represented as Reshape in VPU model
        {"Reshape",                                            LAYER_PARSER(parseReshape)},
        // {"Flatten",                                            LAYER_PARSER(parseReshape)},
        {"Squeeze",                                            LAYER_PARSER(parseReshape)},
        {"Unsqueeze",                                          LAYER_PARSER(parseReshape)},
        {"Crop",                                               LAYER_PARSER(parseCrop)},
        {"Tile",                                               LAYER_PARSER(parseTile)},
        {"NormalizeL2",                                          LAYER_PARSER(parseNormalize)},
        {"PriorBox",                                           LAYER_PARSER(parsePriorBox)},
        {"PriorBoxClustered",                                  LAYER_PARSER(parsePriorBoxClustered)},
        {"Transpose",                                          LAYER_PARSER(parsePermute)},
        {"DetectionOutput",                                    LAYER_PARSER(parseDetectionOutput)},
        {"RegionYolo",                                         LAYER_PARSER(parseRegionYolo)},
        {"ReorgYolo",                                          LAYER_PARSER(parseReorgYolo)},
        {"CTCGreedyDecoder",                                   LAYER_PARSER(parseCTCDecoder)},
        {"Proposal",                                           LAYER_PARSER(parseProposal)},
        {"ROIPooling",                                         LAYER_PARSER(parseROIPooling)},
        {"PSROIPooling",                                       LAYER_PARSER(parsePSROIPooling)},
        {"Interp",                                             LAYER_PARSER(parseInterp)},
        {"Interpolate",                                        LAYER_PARSER(parseInterpolate)},
        // {"Custom",                                             LAYER_PARSER(parseCustom)},
        // {"MTCNN",                                              LAYER_PARSER(parseMTCNN)},
        {"LSTMCell",                                           LAYER_PARSER(parseLSTMCell)},
        {"Pad",                                                LAYER_PARSER(parsePad)},
        {"Resample",                                           LAYER_PARSER(parseResample)},
        {"LSTMSequence",                                       LAYER_PARSER(parseRNN)},
        {"MatMul",                                               LAYER_PARSER(parseGEMM)},
        {"Log",                                                LAYER_PARSER(parseLog)},
        {"Exp",                                                LAYER_PARSER(parseExp)},
        {"ReverseSequence",                                    LAYER_PARSER(parseReverseSequence)},
        {"Gather",                                             LAYER_PARSER(parseGather)},
        {"Floor",                                              LAYER_PARSER(parseFloor)},
        {"TopK",                                               LAYER_PARSER(parseTopK)},
        {"StridedSlice",                                       LAYER_PARSER(parseStridedSlice)},
        {"Select",                                             LAYER_PARSER(parseSelect)},
        {"Erf",                                                LAYER_PARSER(parseErf)},
        {"ExperimentalDetectronDetectionOutput",               LAYER_PARSER(parseExpDetectionOutput)},
        {"ExperimentalDetectronROIFeatureExtractor",           LAYER_PARSER(parseROIFeatureExtractor)},
        {"Convert",                                            LAYER_PARSER(parseConvert)},
        {"ReduceAnd",                                          LAYER_PARSER(parseReduceAnd)},
        {"ReduceMin",                                          LAYER_PARSER(parseReduceMin)},
        {"ReduceMax",                                          LAYER_PARSER(parseReduceMax)},
        {"ReduceSum",                                          LAYER_PARSER(parseReduceSum)},
        {"ReduceMean",                                         LAYER_PARSER(parseReduceMean)},
        // {"TensorIterator",                                     LAYER_PARSER(parseTensorIterator)},
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
        // {"StaticShapeNonMaxSuppression",                       LAYER_PARSER(parseStaticShapeNMS)},
        {"StaticShapeReshape",                                 LAYER_PARSER(parseReshape)},
        {"Mish",                                               LAYER_PARSER(parseMish)},
        {"Gelu",                                               LAYER_PARSER(parseGelu)},
        {"SoftPlus",                                           LAYER_PARSER(parseSoftPlus)},
        {"Swish",                                              LAYER_PARSER(parseSwish)},
        // {"Activation",                                         LAYER_PARSER(parseActivation)},
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
    manager.register_pass<vpu::ConvertMatMulToFC>();
    // ConstantFolding placed here to avoid precision type missmatch when we try to evaluate nodes with BOOL output.
    // For example evaluate_greater_equal calls set_broadcast function with hardcoded BOOL precision.
    // In set_broadcast function we compare original node's precision with hardcoded so we get an error if we change precision before.
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    // manager.register_pass<ngraph::pass::ConvertMatMulToFCorGemm>();
    // ConvertPrecision must be executed before ConvertOpSet1ToLegacy due to this pass works with operations from opsets only
    static const precisions_array precisions = {
        { ngraph::element::i64, ngraph::element::i32 },
        { ngraph::element::u64, ngraph::element::i32 },
        { ngraph::element::u32, ngraph::element::i32 },
        { ngraph::element::boolean, ngraph::element::i32 }
    };
    manager.register_pass<ngraph::pass::ConvertPrecision>(precisions, myriadTypeToFuseMap);

    // manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    //  ConvertOpSet1ToLegacy can produce constants with I64 precision
    // manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ ngraph::element::i64, ngraph::element::i32 }}, myriadTypeToFuseMap);
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
    ngraph::pass::VisualizeTree("/home/akorolev/work/tmp/cnnlayer_to_ngraph/dump_alexnet.svg").run_on_function(nGraphFunc);

    return network;
}

std::set<std::string> FrontEnd::checkSupportedLayers(const ie::CNNNetwork& network) {
    VPU_PROFILE(checkSupportedLayers);

    const auto& env = CompileEnv::get();

    env.log->debug("FrontEnd : Check supported layers");
    VPU_LOGGER_SECTION(env.log);

    std::set<std::string> supportedLayers;

    const auto onSupportedLayer = [&supportedLayers](const NodePtr& node) {
        supportedLayers.insert(node->get_name());
    };

    const auto onUnsupportedLayer = [this](
        const Model& model,
        const NodePtr& node,
        const DataVector& inputs,
        const DataVector& outputs,
        const std::string& /*extraMsg*/) {
        _stageBuilder->addNoneStage(model, node->get_name(), node, inputs, outputs);
    };

    runCommonPasses(cloneNetwork(network), onUnsupportedLayer, onSupportedLayer);

    return supportedLayers;
}

namespace {

std::atomic<int> g_counter(0);

}  // namespace

// std::vector<vpu::CustomLayer::Ptr> getSuitableCustomLayers(const std::vector<vpu::CustomLayer::Ptr>& customLayers,
//                                                            const std::shared_ptr<ngraph::Node>& node) {
//     const auto isSuitableLayer = [&](const vpu::CustomLayer::Ptr& customLayer) {
//         paramVisitor visitor;
//         node->visit_attributes(visitor);
//         auto layerParams = visitor.GetMap();

//         if (!customLayer->meetsWhereRestrictions(layerParams)) {
//             return false;
//         }

//         SizeRuleValidator validator{customLayer, layerParams};
//         for (const auto& kernel : customLayer->kernels()) {
//             kernel->accept(validator);
//             if (!validator.result()) {
//                 return false;
//             }
//         }

//         return true;
//     };

//     auto suitableCustomLayers = std::vector<vpu::CustomLayer::Ptr>{};

//     std::copy_if(begin(customLayers), end(customLayers), back_inserter(suitableCustomLayers), isSuitableLayer);

//     return suitableCustomLayers;
// }

void FrontEnd::parseLayer(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs) {
    parseLayer(model, node, inputs, outputs,
        [this](const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs,
                            const std::string& extraMessage)
        { defaultOnUnsupportedLayerCallback(model, node, inputs, outputs, extraMessage); });
}

void FrontEnd::parseLayer(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs,
                          const FrontEnd::UnsupportedNodeCallback& onUnsupported, const FrontEnd::SupportedNodeCallback& onSupported) {
    // const auto customLayer = _customLayers.find(node->get_type_name());
    // const bool isCustomLayer = customLayer != _customLayers.end() && getSuitableCustomLayer(customLayer->second, node);

    const auto& type = /*isCustomLayer ? "Custom" : */node->get_type_name();
    if (parsers.count(type) == 0) {
        if (onUnsupported) {
            onUnsupported(model, node, inputs, outputs, formatString("unsupported layer type \"%v\"", type));
        }
        return;
    }

    try {
        parsers.at(type)(model, node, inputs, outputs);
        if (onSupported) {
            onSupported(node);
        }
    } catch (const details::UnsupportedLayerException&) {
        throw;
    } catch (const std::exception& error) {
        if (onUnsupported) {
            onUnsupported(model, node, inputs, outputs, error.what());
        }
    }
}

void FrontEnd::processTrivialCases(const Model& model) {
    std::unordered_map<NodePtr, std::pair<Data, Data>> ieDataToTrivialCase;
    for (const auto& data : model->datas()) {
        const auto& origNode = data->origNode();
        if (origNode == nullptr) {
            continue;
        }

        auto& trivialCase = ieDataToTrivialCase[origNode];
        auto& destination = data->usage() == DataUsage::Output ? trivialCase.second : trivialCase.first;
        VPU_THROW_UNLESS(ieDataToTrivialCase.count(origNode) == 0 || destination == nullptr,
            "Encountered node {} which has two vpu data objects {} and {} of the same type {} associated with it, while only one is permitted",
            origNode->get_friendly_name(), destination->name(), data->name(), destination->usage());
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

void FrontEnd::defaultOnUnsupportedLayerCallback(const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs,
                                                 const std::string& extraMessage) {
    const auto& env = CompileEnv::get();
    VPU_THROW_UNSUPPORTED_LAYER_UNLESS(env.config.compileConfig().ignoreUnknownLayers, "Failed to compile layer \"%v\": %v", node->get_friendly_name(), extraMessage);
    _stageBuilder->addNoneStage(model, node->get_friendly_name(), node, inputs, outputs);
}

ModelPtr FrontEnd::runCommonPasses(const ie::CNNNetwork& network) {
    return runCommonPasses(cloneNetwork(network),
        [this](const Model& model, const NodePtr& node, const DataVector& inputs, const DataVector& outputs, const std::string& extraMessage) {
            defaultOnUnsupportedLayerCallback(model, node, inputs, outputs, extraMessage);});
}

ModelPtr FrontEnd::runCommonPasses(ie::CNNNetwork network,
    const UnsupportedNodeCallback& unsupportedLayer, const SupportedNodeCallback& supportedLayer) {
    const auto& env = CompileEnv::get();

    //
    // Clear Front-end state
    //

    _ieParsedNetwork = {};
    _unbatchedOutputs.clear();
    _ieToVpuMap.clear();
    // _customLayers.clear();
    _kernelNodes.clear();
    _lstmWeights.clear();
    _lstmBiases.clear();

    //
    // Parse custom layers
    //

    // if (!env.config.compileConfig().customLayers.empty()) {
    //     env.log->trace("Parse custom layers : %s", env.config.compileConfig().customLayers);
    //     VPU_LOGGER_SECTION(env.log);

    //     if (env.platform != Platform::MYRIAD_X) {
    //         VPU_THROW_FORMAT("Custom layers are not supported for %v platforms", env.platform);
    //     }

    //     _customLayers = CustomLayer::loadFromFile(env.config.compileConfig().customLayers);
    // }

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

        detectNetworkBatch(network, model);

        //
        // Running ngraph passes
        //

        network = convertNetwork(network);

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
        // removeConstLayers(network);
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

        if (!CompileEnv::get().config.compileConfig().disableConvertStages) {
            addDataTypeConvertStages(model);
        }

        addPreProcessStages(model);
    }

    //
    // Parse original layers
    //

    env.log->trace("Parse original nodes");

    DataVector inputs, outputs;
    for (const auto& node : origNodes()) {
        VPU_LOGGER_SECTION(env.log);

        env.log->trace("Try to parse node %s:%s", node->get_name(), node->get_type_name());
        VPU_LOGGER_SECTION(env.log);

        getInputAndOutputData(model, node, inputs, outputs);

        if (env.config.compileConfig().skipAllLayers() || env.config.compileConfig().skipLayerType(node->get_type_name())) {
            _stageBuilder->addNoneStage(model, node->get_name(), node, inputs, outputs);
            supportedLayer(node);
            continue;
        }
        std::cout << "Parse layer with name " << node->get_friendly_name() << " and type " << node->get_type_name() << std::endl;
        parseLayer(model, node, inputs, outputs, unsupportedLayer, supportedLayer);
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

Data FrontEnd::getVpuData(const OutNode& outNode) const {
    IE_ASSERT(outNode.get_node_shared_ptr() != nullptr);

    const auto it = _ieToVpuMap.find(outNode);
    if (it == _ieToVpuMap.end()) {
        return nullptr;
    }

    return it->second;
}

void FrontEnd::bindData(const Data& data, const OutNode& nodeOutput, NodePtr origNode) {
    _ieToVpuMap[nodeOutput] = data;
    data->setOrigNode(origNode);
    data->setOrigOutput(nodeOutput);
}

void FrontEnd::getInputAndOutputData(
        const Model& model,
        const NodePtr& node,
        DataVector& inputs,
        DataVector& outputs) {
    IE_ASSERT(node != nullptr);

    inputs.resize(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto& inputNodeOutput = node->get_input_source_output(i);
        IE_ASSERT(inputNodeOutput.get_node_shared_ptr() != nullptr);
        inputs[i] = getVpuData(inputNodeOutput);
        IE_ASSERT(inputs[i] != nullptr);
    }

    outputs.resize(node->get_output_size());
    for (int i = 0; i < node->get_output_size(); ++i) {
        const auto& outputNode = node->output(i);
        if (const auto data = getVpuData(outputNode)) {
            outputs[i] = data;
        } else {
            const auto& desc = node->get_output_tensor(i);
            DataDesc dataDesc(desc);
            if (dataDesc.type() == DataType::FP32) {
                // To infer the same FP32 models on different devices (CPU, GPU, VPU and so on)
                dataDesc.setType(DataType::FP16);
            }

            // Skip adding data if it not utilized

            // REWROK
            // const bool isNetworkOutput = _ieParsedNetwork.networkOutputs.count(layerOutput->getName()) > 0;
            // const auto isLeaf = getInputTo(layerOutput).empty();
            // if (!isNetworkOutput && isLeaf) {
            //     outputs[i] = nullptr;
            //     continue;
            // }

            outputs[i] = model->addNewData(
                outputNode.get_node_shared_ptr()->get_friendly_name(),
                dataDesc);
            bindData(outputs[i], outputNode, node);
        }
    }
}

ie::Blob::Ptr FrontEnd::shareWeights(const NodePtr& node)  {
    auto constLayer = ngraph::as_type_ptr<ngraph::opset4::Constant>(node);
    if (!constLayer) IE_THROW() << "Cannot share weights! Constant operation is empty!";
    auto dataPrecision = ie::details::convertPrecision(constLayer->get_element_type());

    size_t shapeSize = ngraph::shape_size(constLayer->get_shape());
    size_t byte_size{8};
    if (dataPrecision == ie::Precision::BIN) {
        shapeSize = (shapeSize + (byte_size - 1)) / byte_size;
    }

    ie::TensorDesc td(dataPrecision, {shapeSize}, ie::Layout::C);

    auto blob = make_blob_with_precision(td, std::make_shared<ie::details::ConstAllocatorWrapper>(constLayer));
    blob->allocate();

    return blob;
}

std::tuple<Data, Data> FrontEnd::getWeightsAndBiases(const Model& model, const std::string nodeName,
                                                     const NodePtr& weightsNode, const NodePtr& biasesNode) const {
    auto constant = ngraph::as_type_ptr<ngraph::opset4::Constant>(weightsNode);
    VPU_THROW_UNLESS(constant != nullptr, "Can't get weights. Node with name {} has no constant input", nodeName);
    
    const auto origWeights = shareWeights(weightsNode);
    VPU_THROW_UNLESS(origWeights != nullptr, "Can't get weights. Node with name {} has no constant input", nodeName);

    const auto weights = model->addConstData(
        nodeName + "@weights",
        DataDesc({origWeights->size()}),
        ieBlobContent(origWeights));

    Data biases;
    if (biasesNode != nullptr) {
        auto constBiasesNode = ngraph::as_type_ptr<ngraph::opset4::Constant>(biasesNode);
        VPU_THROW_UNLESS(constBiasesNode != nullptr, "Can't get biases. Node with name {} has no constant input", nodeName);

        const auto origBiases = shareWeights(biasesNode);
        biases = model->addConstData(
            nodeName + "@biases",
            DataDesc({origBiases->size()}),
            ieBlobContent(origBiases));
    } else {
        biases = model->addFakeData();
    }

    return std::make_tuple(weights, biases);
}

}  // namespace vpu
