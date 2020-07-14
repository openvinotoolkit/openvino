// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <set>
#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <ie_metric_helpers.hpp>
#include <ie_util_internal.hpp>
#include <ie_plugin_config.hpp>
#include <network_serializer.h>
#include <threading/ie_executor_manager.hpp>
#include <details/ie_cnn_network_tools.h>

#include <ngraph/ngraph.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>

#include "template_plugin.hpp"
#include "template_executable_network.hpp"
#include "template_pattern_transformation.hpp"

using namespace TemplatePlugin;

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(InferenceEngine::ICNNNetwork&  network,
                                                     const Configuration&           cfg):
    _name(network.getName()),
    _cfg(cfg),
    _waitExecutor(InferenceEngine::ExecutorManager::getInstance()->getExecutor("Template")) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.

    try {
        if (std::shared_ptr<const ngraph::Function> ngraphFunction = network.getFunction()) {
            CompileGraph(ngraphFunction);
        } else {
            THROW_IE_EXCEPTION << "TEMPLATE plugin can compile only IR v10 networks";
        }
    }
    catch (const InferenceEngineException & e) {
        throw e;
    }
    catch (const std::exception & e) {
        THROW_IE_EXCEPTION << "Standard exception from compilation library: " << e.what();
    }
    catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}
// ! [executable_network:ctor_cnnnetwork]

// ! [executable_network:ctor_import_stream]
TemplatePlugin::ExecutableNetwork::ExecutableNetwork(std::istream &                 model,
                                                     const Configuration&           cfg) :
                  _cfg(cfg) {
    // TODO: since Import network is not a mandatory functionality, this ctor can just be removed
}
// ! [executable_network:ctor_import_stream]

// ! [executable_network:compile_graph]
void TemplatePlugin::ExecutableNetwork::CompileGraph(const std::shared_ptr<const ngraph::Function> & ngraphFunction) {
    // TODO: perform actual graph compilation taking `_cfg` into account

    // 1.Copy ngraph::Function first to apply some transformations later in
    // ExecutableNetwork::CompileGraph, which modify original ngraph::Function
    const bool shareConsts = false, constFolding = false;
    std::vector<::ngraph::element::Type> new_types;
    std::vector<::ngraph::PartialShape> new_shapes;

    for (const auto &parameter : ngraphFunction->get_parameters()) {
        new_shapes.emplace_back(parameter->get_partial_shape());
        new_types.emplace_back(parameter->get_element_type());
    }

    auto copyFunction = ngraph::specialize_function(std::const_pointer_cast<ngraph::Function>(ngraphFunction),
        new_types, new_shapes, std::vector<void *>(new_types.size(), nullptr), constFolding, shareConsts);

    // 2. Perform common optimizations and device-specific transformations
    ngraph::pass::Manager passManager;
    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ngraph::pass::CommonOptimizations>();
    // Example: register plugin specific transformation
    passManager.register_pass<ngraph::pass::DecomposeDivideMatcher>();
    passManager.register_pass<ngraph::pass::ReluReluFusionMatcher>();
    // Register any other transformations
    // ..

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device hardware-dependent graph
    passManager.run_passes(copyFunction);

    // 3. Iterate over operations and create hardware-specific ngraph
    for (const auto& op : copyFunction->get_ordered_ops()) {
        // TODO: map ngraph `op` to device operation
    }

    // 4. Perform any other steps like allocation and filling device buffers, and so on
}
// ! [executable_network:compile_graph]

// ! [executable_network:create_infer_request_impl]
InferenceEngine::InferRequestInternal::Ptr TemplatePlugin::ExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                     InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<TemplateInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}
// ! [executable_network:create_infer_request_impl]

// ! [executable_network:create_infer_request]
void TemplatePlugin::ExecutableNetwork::CreateInferRequest(IInferRequest::Ptr& asyncRequest) {
    auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    auto asyncThreadSafeImpl = std::make_shared<TemplateAsyncInferRequest>(std::static_pointer_cast<TemplateInferRequest>(internalRequest),
                                                                           _taskExecutor, _waitExecutor, _callbackExecutor);
    asyncRequest.reset(new InferenceEngine::InferRequestBase<TemplateAsyncInferRequest>(asyncThreadSafeImpl),
                       [](InferenceEngine::IInferRequest *p) { p->Release(); });
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
}
// ! [executable_network:create_infer_request]

// ! [executable_network:get_config]
void TemplatePlugin::ExecutableNetwork::GetConfig(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    // TODO: return more supported values for config keys
    if (name == CONFIG_KEY(DEVICE_ID) ||
        name == CONFIG_KEY(PERF_COUNT)) {
        result = _cfg.Get(name);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}
// ! [executable_network:get_config]

// ! [executable_network:get_metric]
void TemplatePlugin::ExecutableNetwork::GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *) const {
    // TODO: return more supported values for metrics
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            CONFIG_KEY(DEVICE_ID),
            CONFIG_KEY(PERF_COUNT)});
    } else if (METRIC_KEY(NETWORK_NAME) == name) {
        result = IE_SET_METRIC(NETWORK_NAME, _name);
    } else if (METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        // TODO: fill with actual number
        unsigned int value = 1;
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}
// ! [executable_network:get_metric]

// ! [executable_network:export_impl]
void TemplatePlugin::ExecutableNetwork::ExportImpl(std::ostream& dlaModel) {
    // TODO: Code which exports graph from std::ostream
}
// ! [executable_network:export_impl]
