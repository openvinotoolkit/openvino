// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "mkldnn_plugin.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_weights_cache.hpp"
#include "mkldnn_itt.h"

#include <legacy/net_pass.h>
#include <threading/ie_executor_manager.hpp>
#include <memory>
#include <ie_plugin_config.hpp>
#include <vector>
#include <tuple>
#include <ie_system_conf.h>
#include <generic_ie.hpp>
#include <nodes/list.hpp>
#include <legacy/ie_util_internal.hpp>
#include <legacy/graph_transformer.h>
#include <legacy/ie_ngraph_utils.hpp>

#include <legacy/convert_function_to_cnn_network.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>
#include <transformations/convert_opset2_to_opset1/convert_opset2_to_opset1.hpp>
#include <transformations/convert_opset3_to_opset2/convert_opset3_to_opset2.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/tensor_iterator_transformations/apply_transformations_to_ti_body.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_ops/fully_connected.hpp"

#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
#if defined(_WIN32) || defined(WIN32)
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>

#endif
#endif

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

Engine::Engine() {
    _pluginName = "CPU";
    extensionManager->AddExtension(std::make_shared<Extensions::Cpu::MKLDNNExtensions>());
}

Engine::~Engine() {
    ExecutorManager::getInstance()->clear("CPUStreamsExecutor");
    ExecutorManager::getInstance()->clear("CPUCallbackExecutor");
}

static void Transformation(ICNNNetwork::Ptr& clonedNetwork) {
    OV_ITT_SCOPED_TASK(MKLDNNPlugin::itt::domains::MKLDNNPlugin, "Transformation");

    const auto transformations_callback = [](const std::shared_ptr<const ::ngraph::Node> &node) -> bool {
        // DepthToSpace node implementation supports only equal input/output tensors with rank <= 5
        if (auto dtsOp = std::dynamic_pointer_cast<const ::ngraph::opset3::DepthToSpace>(node)) {
            return dtsOp->input_value(0).get_shape().size() <= 5lu && dtsOp->input_value(0).get_shape().size() == dtsOp->get_output_shape(0).size();
        }

        // SpaceToDepth node implementation supports only equal input/output tensors with rank <= 5
        if (auto stdOp = std::dynamic_pointer_cast<const ::ngraph::opset3::SpaceToDepth>(node)) {
            return stdOp->input_value(0).get_shape().size() <= 5lu && stdOp->input_value(0).get_shape().size() == stdOp->get_output_shape(0).size();
        }

        if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
            return fc_op->input_value(0).get_shape().size() == 3ul;
        }

        return std::dynamic_pointer_cast<const ngraph::opset2::Gelu>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset2::BatchToSpace>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset2::SpaceToBatch>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset3::ExtractImagePatches>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset4::HSwish>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset4::ReduceL1>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset4::ReduceL2>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset4::SoftPlus>(node) ||
               std::dynamic_pointer_cast<const ngraph::opset4::Pad>(node);
    };
    auto nGraphFunc = clonedNetwork->getFunction();
    // Disable shape inference (WA for generic operations)
    ngraph::op::GenericIE::DisableReshape noReshape(nGraphFunc);

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
    manager.register_pass<ngraph::pass::ConvertPriorBox>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();

    std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> convert_precision_list {
            {ngraph::element::i64, ngraph::element::i32},
            {ngraph::element::u64, ngraph::element::i32},
            {ngraph::element::u16, ngraph::element::i32},
            {ngraph::element::u32, ngraph::element::i32},
            {ngraph::element::f16, ngraph::element::f32},
            {ngraph::element::boolean, ngraph::element::u8},
    };

    for (auto & precision : convert_precision_list) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(precision.first, precision.second);
    }

    manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);

    manager.set_callback(transformations_callback);
    manager.run_passes(nGraphFunc);

    // Apply all transformations to TensorIterator body
    ngraph::pass::Manager ti_manager;
    ti_manager.register_pass<ngraph::pass::ApplyTransformationsToTIBody>(manager);
    ti_manager.run_passes(nGraphFunc);

    clonedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(nGraphFunc, *clonedNetwork);

    // WA: after conversion to CNNNetwork user precision can redefine input/output precisions
    // so we need to apply additional precision conversion but only for inputs and outputs
    for (auto & precision : convert_precision_list) {
        NetPass::ConvertIOPrecision(*clonedNetwork, convertPrecision(precision.first), convertPrecision(precision.second));
    }
}

InferenceEngine::ExecutableNetworkInternal::Ptr
Engine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork &network, const std::map<std::string, std::string> &config) {
    OV_ITT_SCOPED_TASK(itt::domains::MKLDNNPlugin, "Engine::LoadExeNetworkImpl");

    // verification of supported input
    InferenceEngine::InputsDataMap _networkInputs;
    network.getInputsInfo(_networkInputs);
    for (const auto &ii : _networkInputs) {
        auto input_precision = ii.second->getPrecision();
        if (input_precision != InferenceEngine::Precision::FP32 &&
            input_precision != InferenceEngine::Precision::I32 &&
            input_precision != InferenceEngine::Precision::U16 &&
            input_precision != InferenceEngine::Precision::I16 &&
            input_precision != InferenceEngine::Precision::I8 &&
            input_precision != InferenceEngine::Precision::U8 &&
            input_precision != InferenceEngine::Precision::BOOL) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str
                               << "Input image format " << input_precision << " is not supported yet...";
        }
    }

    // TODO: handle input precision differently - per input and not one per network...

    // TODO: Clarify the behavior of SetConfig method. Skip eng_config or not?
    Config conf = engConfig;
    conf.readProperties(config);

    if (conf.enableDynamicBatch) {
        conf.batchLimit = static_cast<int>(network.getBatchSize());
    }

    std::shared_ptr<ICNNNetwork> clonedNetwork = cloneNetwork(network);
    bool is_transformed = false;
    if (clonedNetwork->getFunction()) {
        Transformation(clonedNetwork);
        is_transformed = true;
    }
    auto implNetwork = std::dynamic_pointer_cast<details::CNNNetworkImpl>(clonedNetwork);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        ConstTransformer transformator(implNetwork.get());
        transformator.fullTrim();
        if (!is_transformed) {
            NetPass::ConvertPrecision(*implNetwork, Precision::I64, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::U64, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::U32, Precision::I32);
            NetPass::ConvertPrecision(*implNetwork, Precision::FP16, Precision::FP32);
            NetPass::ConvertPrecision(*implNetwork, Precision::BOOL, Precision::U8);
            NetPass::ConvertPrecision(*implNetwork, Precision::U16, Precision::I32);
        }
    }

    return std::make_shared<MKLDNNExecNetwork>(*clonedNetwork, conf, extensionManager, weightsSharing);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    // accumulate config parameters on engine level
    engConfig.readProperties(config);
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    Parameter result;
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key " << name;
    }
    return result;
}

static bool hasAVX512() {
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
    unsigned int regs[4] = {7, 0, 0, 0};
#if defined(_WIN32) || defined(WIN32)
    __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
    __cpuid_count(regs[0], regs[1], regs[0], regs[1], regs[2], regs[3]);
#endif
    if (regs[1] & (1U << 16))
        return true;
#endif
    return false;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS));
        metrics.push_back(METRIC_KEY(RANGE_FOR_STREAMS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string brand_string;
#if !defined(__arm__) && !defined(_M_ARM) && !defined(__aarch64__) && !defined(_M_ARM64)
        unsigned int addr_list[3] = { 0x80000002, 0x80000003, 0x80000004 };
        unsigned int regs[4];
        for (auto addr : addr_list) {
            regs[0] = addr;
#if defined(_WIN32) || defined(WIN32)
            __cpuid(reinterpret_cast<int*>(regs), regs[0]);
#else
            __get_cpuid(regs[0], &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
            char *ch = reinterpret_cast<char*>(&regs[0]);
            for (size_t j = 0; j < sizeof(regs); j++)
                brand_string += ch[j];
        }
#else
        brand_string = "Non Intel Architecture";
#endif
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, brand_string);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities;
        if (with_cpu_x86_bfloat16())
            capabilities.push_back(METRIC_VALUE(BF16));
        if (hasAVX512())
            capabilities.push_back(METRIC_VALUE(WINOGRAD));
        capabilities.push_back(METRIC_VALUE(FP32));
        capabilities.push_back(METRIC_VALUE(FP16));
        capabilities.push_back(METRIC_VALUE(INT8));
        capabilities.push_back(METRIC_VALUE(BIN));
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && opt : engConfig._config)
            configKeys.push_back(opt.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 1, 1);
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, range);
    } else if (name == METRIC_KEY(RANGE_FOR_STREAMS)) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, parallel_get_max_threads());
        IE_SET_METRIC_RETURN(RANGE_FOR_STREAMS, range);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    extensionManager->AddExtension(extension);
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config, QueryNetworkResult& res) const {
    MKLDNNWeightsSharing::Ptr fake_w_cache;
    auto function = network.getFunction();
    if (function != nullptr) {
        std::unordered_set<std::string> originalOps;
        for (auto&& node : function->get_ops()) {
            originalOps.emplace(node->get_friendly_name());
        }
        auto clonedNetwork = cloneNetwork(network);
        Transformation(clonedNetwork);
        std::unordered_set<std::string> supported;
        std::unordered_set<std::string> unsupported;
        for (details::CNNNetworkIterator itLayer{clonedNetwork.get()}; itLayer != details::CNNNetworkIterator(); itLayer++) {
            auto layerIsSupported = [&] {
                std::unique_ptr<MKLDNNNode> ptr;
                try {
                    ptr.reset(MKLDNNNode::CreateNode(*itLayer, {mkldnn::engine::kind::cpu, 0}, extensionManager, fake_w_cache));
                } catch (InferenceEngine::details::InferenceEngineException&) {
                     return false;
                }
                return true;
            } ();
            for (auto&& fusedLayerName : ngraph::getFusedNamesVector((*itLayer)->getNode())) {
                if (contains(originalOps, fusedLayerName)) {
                    if (layerIsSupported) {
                        supported.emplace(fusedLayerName);
                    } else {
                        unsupported.emplace(fusedLayerName);
                    }
                }
            }
        }

        for (auto&& node : function->get_ops()) {
            if (!contains(unsupported, node->get_friendly_name())) {
                for (auto&& inputNodeOutput : node->input_values()) {
                    if (ngraph::op::is_constant(inputNodeOutput.get_node())) {
                        supported.emplace(inputNodeOutput.get_node()->get_friendly_name());
                    }
                }
                for (auto&& outputs : node->outputs()) {
                    for (auto&& outputNodeInput : outputs.get_target_inputs()) {
                        if (ngraph::op::is_output(outputNodeInput.get_node())) {
                            supported.emplace(outputNodeInput.get_node()->get_friendly_name());
                        }
                    }
                }
            }
        }

        for (auto&& layerName : supported) {
            if (!contains(unsupported, layerName)) {
                res.supportedLayersMap.emplace(layerName, GetName());
            }
        }
    } else {
        details::CNNNetworkIterator i(&network);
        while (i != details::CNNNetworkIterator()) {
            try {
                mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
                // if we can create and have not thrown exception, then layer is supported
                std::unique_ptr <MKLDNNNode>(MKLDNNNode::CreateNode(*i, eng, extensionManager, fake_w_cache));
                res.supportedLayersMap.insert({ (*i)->name, GetName() });
            } catch (InferenceEngine::details::InferenceEngineException&) {
            }
            i++;
        }
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MKLDNNPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, version)
