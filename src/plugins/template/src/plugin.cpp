// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/property_supervisor.hpp"
#include "template/config.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/template_pattern_transformation.hpp"

using namespace TemplatePlugin;

namespace {
static constexpr const char* wait_executor_name = "TemplateWaitExecutor";
static constexpr const char* stream_executor_name = "TemplateStreamsExecutor";
}  // namespace

// ! [plugin:ctor]
Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("TEMPLATE");

    // create ngraph backend which performs inference using ngraph reference implementations
    _backend = ngraph::runtime::Backend::create();

    // create default stream executor with a given name
    _waitExecutor = get_executor_manager()->getIdleCPUStreamsExecutor({wait_executor_name});

    // Add plugin specific properties
    m_properties.set_name(get_device_name())
        .add(ov::common_property(ov::device::architecture), get_device_name())
        .add(ov::common_property(ov::device::capabilities),
             {ov::device::capability::EXPORT_IMPORT, ov::device::capability::FP32})
        .add(ov::common_property(ov::range_for_async_infer_requests), std::make_tuple(uint{1}, uint{1}, uint{1}));

    // Add common read write properties used in template plugin and template compiled model
    m_properties.add(rw_properties.m_properties);

    // If plugin has several devices we can add property for each device
    for (auto device_id : {"0"}) {
        m_properties.add(device_id, ov::PropertySupervisor{}.add(ov::device::full_name, "Template Device Full Name"));
    }
}
// ! [plugin:ctor]

// ! [plugin:dtor]
Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    get_executor_manager()->clear(stream_executor_name);
    get_executor_manager()->clear(wait_executor_name);
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
    passManager.get_pass_config()->disable<ov::pass::UnrollIf>();
    // This transformation changes output name
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

    auto fullConfig = m_properties.merge(properties);
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(rw_properties._streamsExecutorConfig);
    streamsExecutorConfig._name = stream_executor_name;
    auto compiled_model =
        std::make_shared<CompiledModel>(model->clone(),
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

    auto fullConfig = rw_properties;
    auto config = fullConfig.m_properties.merge(properties);
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
    streamsExecutorConfig._name = stream_executor_name;
    auto compiled_model =
        std::make_shared<CompiledModel>(ov_model,
                                        shared_from_this(),
                                        get_executor_manager()->getIdleCPUStreamsExecutor(streamsExecutorConfig),
                                        config);
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

    auto fullConfig = m_properties.merge(properties);

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto supported = InferenceEngine::GetSupportedNodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            // 1. It is needed to apply all transformations as it is done in compile_model
            transform_model(model);
        },
        [&](std::shared_ptr<ngraph::Node> node) {
            // 2. Сheck whether node is supported
            ngraph::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
        // clang-format off
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
        // clang-format on
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

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
// ! [plugin:create_plugin_engine]
