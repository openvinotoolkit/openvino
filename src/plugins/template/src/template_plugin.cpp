// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "template_plugin.hpp"

#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_metric_helpers.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "template/template_config.hpp"
#include "template_compiled_model.hpp"
#include "template_infer_request.hpp"
#include "template_itt.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/template_pattern_transformation.hpp"

using namespace TemplatePlugin;

// ! [plugin:ctor]
Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("TEMPLATE");

    // create ngraph backend which performs inference using ngraph reference implementations
    _backend = ngraph::runtime::Backend::create();

    // create default stream executor with a given name
    _waitExecutor = get_executor_manager()->getIdleCPUStreamsExecutor({"TemplateWaitExecutor"});
}
// ! [plugin:ctor]

// ! [plugin:dtor]
Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    get_executor_manager()->clear("TemplateStreamsExecutor");
    get_executor_manager()->clear("TemplateWaitExecutor");
    // NOTE: Uncomment this if Inference Engine Executor cache is used to create callback executor
    // executorManager()->clear("TemplateCallbackExecutor");
}
// ! [plugin:dtor]

ov::RemoteContext TemplatePlugin::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext TemplatePlugin::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

// ! [plugin:transform_network]
void transform_model(const std::shared_ptr<ov::Model>& model) {
    // Perform common optimizations and device-specific transformations
    ov::pass::Manager passManager;
    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ov::pass::CommonOptimizations>();
    passManager.get_pass_config()->disable<ov::pass::ConvertReduceSumToPooling>();
    // Example: register plugin specific transformation
    passManager.register_pass<ov::pass::DecomposeDivideMatcher>();
    passManager.register_pass<ov::pass::ReluReluFusionMatcher>();
    // Register any other transformations
    // ..

    const auto& pass_config = passManager.get_pass_config();

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(model);
}
// ! [plugin:transform_network]

// ! [plugin:load_exe_network_impl]
std::shared_ptr<ov::ICompiledModel> TemplatePlugin::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                          const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::compile_model");

    auto fullConfig = Configuration{properties, _cfg};
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(fullConfig._streamsExecutorConfig);
    streamsExecutorConfig._name = "TemplateStreamsExecutor";
    auto compiled_model =
        std::make_shared<CompiledModel>(model,
                                        shared_from_this(),
                                        get_executor_manager()->getIdleCPUStreamsExecutor(streamsExecutorConfig),
                                        fullConfig);
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> TemplatePlugin::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                          const ov::AnyMap& properties,
                                                                          const ov::RemoteContext& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}
// ! [plugin:load_exe_network_impl]

// ! [plugin:import_network]
std::shared_ptr<ov::ICompiledModel> TemplatePlugin::Plugin::import_model(std::istream& model,
                                                                         const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::import_model");

    auto fullConfig = Configuration{properties, _cfg};
    // read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // read blob content
    ov::Tensor weights;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        weights = ov::Tensor(ov::element::from<char>(), ov::Shape{dataSize});
        model.read(weights.data<char>(), dataSize);
    }

    auto ov_model = get_core()->read_model(xmlString, weights);
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(fullConfig._streamsExecutorConfig);
    streamsExecutorConfig._name = "TemplateStreamsExecutor";
    auto compiled_model =
        std::make_shared<CompiledModel>(ov_model,
                                        shared_from_this(),
                                        get_executor_manager()->getIdleCPUStreamsExecutor(streamsExecutorConfig),
                                        fullConfig);
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> TemplatePlugin::Plugin::import_model(std::istream& model,
                                                                         const ov::RemoteContext& context,
                                                                         const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}
// ! [plugin:import_network]

// ! [plugin:query_network]
ov::SupportedOpsMap TemplatePlugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                        const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::query_model");

    Configuration fullConfig{properties, _cfg, false};

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto supported = InferenceEngine::GetSupportedNodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            // 1. It is needed to apply all transformations as it is done in LoadExeNetworkImpl
            transform_model(model);
        },
        [&](std::shared_ptr<ngraph::Node> node) {
            // 2. Сheck whether node is supported
            ngraph::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#undef _OPENVINO_OP_REG
            return op_super_set.contains_type(node->get_type_info());
        });

    // 3. Produce the result
    ov::SupportedOpsMap res;
    for (auto&& layerName : supported) {
        res.emplace(layerName, get_device_name());
    }

    return res;
}
// ! [plugin:query_network]

// ! [plugin:set_config]
void TemplatePlugin::Plugin::set_property(const ov::AnyMap& properties) {
    _cfg = Configuration{properties, _cfg};
}
// ! [plugin:set_config]

// ! [plugin:get_config]
ov::Any TemplatePlugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        std::vector<std::string> supportedMetrics = {METRIC_KEY(AVAILABLE_DEVICES),
                                                     METRIC_KEY(SUPPORTED_METRICS),
                                                     METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                     METRIC_KEY(FULL_DEVICE_NAME),
                                                     METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                     METRIC_KEY(DEVICE_ARCHITECTURE),
                                                     METRIC_KEY(OPTIMIZATION_CAPABILITIES),
                                                     METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)};
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, supportedMetrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {CONFIG_KEY(DEVICE_ID),
                                               CONFIG_KEY(PERF_COUNT),
                                               ov::hint::performance_mode.name(),
                                               TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
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
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, std::make_tuple(uint{1}, uint{1}, uint{1}));
    } else {
        return _cfg.Get(name);
    }
}
// ! [plugin:get_config]

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
// ! [plugin:create_plugin_engine]
