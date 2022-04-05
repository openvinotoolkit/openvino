// Copyright (C) 2018-2022 Intel Corporation
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

#include "openvino/core/except.hpp"
#include "template/template_config.hpp"
#include "template_itt.hpp"
#include "template_plugin.hpp"
#include "template_executable_network.hpp"
#include "template_infer_request.hpp"
#include "transformations/template_pattern_transformation.hpp"
#include "transformations/preprocessing/preprocessing.hpp"

#include "cpp_interfaces/interface/internal_properties.hpp"
// clang-format on

using namespace TemplatePlugin;

// ! [plugin:ctor]
Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    _pluginName = "TEMPLATE";

    // create ngraph backend which performs inference using ngraph reference implementations
    _backend = ngraph::runtime::Backend::create();

    // create default stream executor with a given name
    _waitExecutor = executorManager()->getIdleCPUStreamsExecutor({"TemplateWaitExecutor"});

    // Add plugin specific properties
    _properties.set_name(_pluginName)
        .add(ov::common_property(ov::device::architecture), "TEMPLATE")
        .add(ov::common_property(ov::device::capabilities),
             {ov::device::capability::EXPORT_IMPORT, ov::device::capability::FP32})
        .add(ov::common_property(ov::range_for_async_infer_requests), std::make_tuple(uint{1}, uint{1}, uint{1}));

    // Add common read write properties used in template plugin and template compiled model
    _properties.add(_cfg._properties);

    // If plugin has several devices we can add property for each device
    for (auto device_id : {"0"}) {
        _properties.add(device_id, ov::PropertyAccess{}.add(ov::device::full_name, "TEMPLATE_DEVICE_0"));
    }
}
// ! [plugin:ctor]

// ! [plugin:dtor]
Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    executorManager()->clear("TemplateStreamsExecutor");
    executorManager()->clear("TemplateWaitExecutor");
    // NOTE: Uncomment this if Inference Engine Executor cache is used to create callback executor
    // executorManager()->clear("TemplateCallbackExecutor");
}
// ! [plugin:dtor]

// ! [plugin:transform_network]

std::shared_ptr<ngraph::Function> TransformNetwork(const std::shared_ptr<const ngraph::Function>& function,
                                                   const InferenceEngine::InputsDataMap& inputInfoMap,
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
    // G-API supports only FP32 networks for pre-processing
    bool needF16toF32 = false;
    for (const auto& param : function->get_parameters()) {
        if (param->get_element_type() == ngraph::element::f16 &&
            inputInfoMap.at(param->get_friendly_name())->getTensorDesc().getPrecision() !=
                InferenceEngine::Precision::FP16) {
            needF16toF32 = true;
            break;
        }
    }
    if (needF16toF32) {
        passManager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::f16, ngraph::element::f32}});
    }
    // Example: register plugin specific transformation
    passManager.register_pass<ov::pass::DecomposeDivideMatcher>();
    passManager.register_pass<ov::pass::ReluReluFusionMatcher>();
    // Register any other transformations
    // ..

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(transformedNetwork);

    return transformedNetwork;
}
// ! [plugin:transform_network]

// ! [plugin:load_exe_network_impl]
InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::LoadExeNetworkImpl(
    const InferenceEngine::CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::LoadExeNetworkImpl");

    InferenceEngine::InputsDataMap networkInputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputs = network.getOutputsInfo();

    return std::make_shared<ExecutableNetwork>(network.getFunction(),
                                               networkInputs,
                                               networkOutputs,
                                               _properties.merge(config),
                                               std::static_pointer_cast<Plugin>(shared_from_this()));
}
// ! [plugin:load_exe_network_impl]

// ! [plugin:import_network]
InferenceEngine::IExecutableNetworkInternal::Ptr Plugin::ImportNetwork(
    std::istream& modelStream,
    const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::ImportNetwork");

    auto exec = std::make_shared<ExecutableNetwork>(modelStream,
                                                    _properties.merge(config),
                                                    std::static_pointer_cast<Plugin>(shared_from_this()));
    SetExeNetworkInfo(exec, exec->_function);
    return exec;
}
// ! [plugin:import_network]

// ! [plugin:query_network]
InferenceEngine::QueryNetworkResult Plugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                         const std::map<std::string, std::string>& config) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::QueryNetwork");

    auto fullConfig = _properties.merge(config);
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
    ngraph::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    for (auto&& node : transformedFunction->get_ops()) {
        // Extract transformation history from transformed node as list of nodes
        for (auto&& fusedLayerName : ngraph::getFusedNamesVector(node)) {
            // Filter just nodes from original operation set
            // TODO: fill with actual decision rules based on whether kernel is supported by backend
            if (InferenceEngine::details::contains(originalOps, fusedLayerName)) {
                if (op_super_set.contains_type(friendlyNameToType[fusedLayerName])) {
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
                if (ngraph::op::is_constant(inputNodeOutput.get_node()) ||
                    ngraph::op::is_parameter(inputNodeOutput.get_node())) {
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
            if (!InferenceEngine::details::contains(
                    supported,
                    node->output(0).get_target_inputs().begin()->get_node()->get_friendly_name())) {
                supported.erase(node->get_friendly_name());
            }
        } else if (ngraph::op::is_output(node)) {
            if (!InferenceEngine::details::contains(supported,
                                                    node->input_values().begin()->get_node()->get_friendly_name())) {
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

// ! [plugin:create_plugin_engine]
static const InferenceEngine::Version version = {{2, 1}, CI_BUILD_NUMBER, "openvino_template_plugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)
// ! [plugin:create_plugin_engine]
