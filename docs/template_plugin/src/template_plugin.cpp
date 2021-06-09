// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <ie_algorithm.hpp>

#include <threading/ie_executor_manager.hpp>

#include <ngraph/op/util/op_types.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/opsets/opset.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/rt_info/fused_names_attribute.hpp>
#include <transformations/convert_precision.hpp>

#include "template/template_config.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"
#include "template_executable_network.hpp"
#include "template_infer_request.hpp"
#include "transformations/template_pattern_transformation.hpp"
#include "transformations/preprocessing/preprocessing.hpp"
// clang-format on

using namespace TemplatePlugin;

// ! [plugin:ctor]
Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    _pluginName = "TEMPLATE";

    // create ngraph backend which performs inference using ngraph reference implementations
    ngraph::runtime::Backend::set_backend_shared_library_search_directory("");
    _backend = ngraph::runtime::Backend::create("INTERPRETER");

    // create default stream executor with a given name
    _waitExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({"TemplateWaitExecutor"});
}
// ! [plugin:ctor]

// ! [plugin:dtor]
Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    InferenceEngine::ExecutorManager::getInstance()->clear("TemplateStreamsExecutor");
    InferenceEngine::ExecutorManager::getInstance()->clear("TemplateWaitExecutor");
    // NOTE: Uncomment this if Inference Engine Executor cache is used to create callback executor
    // ExecutorManager::getInstance()->clear("TemplateCallbackExecutor");
}
// ! [plugin:dtor]

// ! [plugin:transform_network]

std::shared_ptr<ngraph::Function> TransformNetwork(const std::shared_ptr<const ngraph::Function>& function, const InferenceEngine::InputsDataMap& inputInfoMap,
                                                   const InferenceEngine::OutputsDataMap& outputsInfoMap) {
    // 1. Copy ngraph::Function first to apply some transformations which modify original ngraph::Function
    auto transformedNetwork = ngraph::clone_function(*function);

    // 2. Perform common optimizations and device-specific transformations
    ngraph::pass::Manager passManager;
    // Example: register transformation to convert preprocessing information to graph nodes
    passManager.register_pass<ngraph::pass::AddPreprocessing>(inputInfoMap);
    // TODO: add post-processing based on outputsInfoMap
    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ngraph::pass::CommonOptimizations>();
    // Template plugin handles only FP32 networks
    passManager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array {{ngraph::element::f16, ngraph::element::f32}});
    // Example: register plugin specific transformation
    passManager.register_pass<ngraph::pass::DecomposeDivideMatcher>();
    passManager.register_pass<ngraph::pass::ReluReluFusionMatcher>();
    // Register any other transformations
    // ..

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(transformedNetwork);

    return transformedNetwork;
}
// ! [plugin:transform_network]

// ! [plugin:load_exe_network_impl]
InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network, const ConfigMap& config) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::LoadExeNetworkImpl");

    InferenceEngine::InputsDataMap networkInputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputs = network.getOutputsInfo();

    auto fullConfig = Configuration {config, _cfg};
    return std::make_shared<ExecutableNetwork>(network.getFunction(), networkInputs, networkOutputs, fullConfig,
                                               std::static_pointer_cast<Plugin>(shared_from_this()));
}
// ! [plugin:load_exe_network_impl]

// ! [plugin:import_network]
InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::ImportNetwork(std::istream& modelStream, const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::ImportNetwork");

    auto fullConfig = Configuration {config, _cfg};
    return std::make_shared<ExecutableNetwork>(modelStream, fullConfig, std::static_pointer_cast<Plugin>(shared_from_this()));
}
// ! [plugin:import_network]

// ! [plugin:query_network]
InferenceEngine::QueryNetworkResult Plugin::QueryNetwork(const InferenceEngine::CNNNetwork& network, const ConfigMap& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::QueryNetwork");

    Configuration fullConfig {config, _cfg, false};
    auto function = network.getFunction();

    // 1. First of all we should store initial input operation set
    std::unordered_set<std::string> originalOps;
    std::map<std::string, ngraph::NodeTypeInfo> friendlyNameToType;
    for (auto&& node : function->get_ops()) {
        originalOps.emplace(node->get_friendly_name());
        friendlyNameToType[node->get_friendly_name()] = node->get_type_info();
    }

    // 2. It is needed to apply all transformations as it is done in LoadExeNetworkImpl
    auto transformedFunction = TransformNetwork(function, network.getInputsInfo(), network.getOutputsInfo());

    // 3. The same input node can be transformed into supported and unsupported backend node
    // So we need store as supported either unsupported node sets
    std::unordered_set<std::string> supported;
    std::unordered_set<std::string> unsupported;
    auto opset = ngraph::get_opset4();
    for (auto&& node : transformedFunction->get_ops()) {
        // Extract transformation history from transformed node as list of nodes
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(node)) {
            // Filter just nodes from original operation set
            // TODO: fill with actual decision rules based on whether kernel is supported by backend
            if (InferenceEngine::details::contains(originalOps, fusedLayerName)) {
                if (opset.contains_type(friendlyNameToType[fusedLayerName])) {
                    supported.emplace(fusedLayerName);
                } else {
                    unsupported.emplace(fusedLayerName);
                }
            }
        }
    }

    // 4. The result set should contain just nodes from supported set
    for (auto&& unsupportedNode : unsupported) {
        supported.erase(unsupportedNode);
    }

    for (auto&& node : function->get_ops()) {
        // 5. If some housekeeping nodes were not added - add them.
        if (InferenceEngine::details::contains(supported, node->get_friendly_name())) {
            for (auto&& inputNodeOutput : node->input_values()) {
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) || ngraph::op::is_parameter(inputNodeOutput.get_node())) {
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

        // 6. Eliminate subgraphs that consist of housekeeping nodes only
        if (ngraph::op::is_constant(node) || ngraph::op::is_parameter(node)) {
            if (!InferenceEngine::details::contains(supported, node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        } else if (ngraph::op::is_output(node)) {
            if (!InferenceEngine::details::contains(supported, node->input_values().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        }
    }

    // 7. Produce the result
    InferenceEngine::QueryNetworkResult res;
    for (auto&& layerName : supported) {
        res.supportedLayersMap.emplace(layerName, GetName());
    }

    return res;
}
// ! [plugin:query_network]

// ! [plugin:add_extension]
void Plugin::AddExtension(const InferenceEngine::IExtensionPtr& /*extension*/) {
    // TODO: add extensions if plugin supports extensions
    IE_THROW(NotImplemented);
}
// ! [plugin:add_extension]

// ! [plugin:set_config]
void Plugin::SetConfig(const ConfigMap& config) {
    _cfg = Configuration {config, _cfg};
}
// ! [plugin:set_config]

// ! [plugin:get_config]
InferenceEngine::Parameter Plugin::GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& /*options*/) const {
    return _cfg.Get(name);
}
// ! [plugin:get_config]

// ! [plugin:get_metric]
InferenceEngine::Parameter Plugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {METRIC_KEY(AVAILABLE_DEVICES),         METRIC_KEY(SUPPORTED_METRICS),
                                                     METRIC_KEY(SUPPORTED_CONFIG_KEYS),     METRIC_KEY(FULL_DEVICE_NAME),
                                                     METRIC_KEY(IMPORT_EXPORT_SUPPORT),     METRIC_KEY(DEVICE_ARCHITECTURE),
                                                     METRIC_KEY(OPTIMIZATION_CAPABILITIES), METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)};
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {CONFIG_KEY(DEVICE_ID), CONFIG_KEY(PERF_COUNT), TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config {}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            if (configKey != InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS) {
                configKeys.emplace_back(configKey);
            }
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (METRIC_KEY(AVAILABLE_DEVICES) == name) {
        // TODO: fill list of available devices
        std::vector<std::string> availableDevices = {""};
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        std::string name = "Template Device Full Name";
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, name);
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (METRIC_KEY(DEVICE_ARCHITECTURE) == name) {
        // TODO: return device architecture for device specified by DEVICE_ID config
        std::string arch = "TEMPLATE";
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, arch);
    } else if (METRIC_KEY(OPTIMIZATION_CAPABILITIES) == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32
        std::vector<std::string> capabilities = {METRIC_VALUE(FP32) /*, TEMPLATE_METRIC_VALUE(HARDWARE_CONVOLUTION)*/};
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS) == name) {
        // TODO: fill with actual values
        using uint = unsigned int;
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, std::make_tuple(uint {1}, uint {1}, uint {1}));
    } else {
        IE_THROW(NotFound) << "Unsupported device metric: " << name;
    }
}
// ! [plugin:get_metric]

// ! [plugin:create_plugin_engine]
static const InferenceEngine::Version version = {{2, 1}, CI_BUILD_NUMBER, "templatePlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
// ! [plugin:create_plugin_engine]
