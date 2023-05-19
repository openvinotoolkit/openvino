// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Inference Engine plugin API wrapper, to be used by particular implementors
 * @file ie_iplugin_internal.hpp
 */

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"

#include <fstream>
#include <istream>
#include <map>
#include <memory>
#include <openvino/runtime/remote_context.hpp>
#include <string>
#include <transformations/common_optimizations/fused_names_cleanup.hpp>
#include <unordered_set>

#include "any_copy.hpp"
#include "blob_factory.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_cnn_network.h"
#include "dev/converter_utils.hpp"
#include "exec_graph_info.hpp"
#include "ie_algorithm.hpp"
#include "ie_api.h"
#include "ie_icore.hpp"
#include "ie_iextension.h"
#include "ie_input_info.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_parameter.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "threading/ie_executor_manager.hpp"
#include "transformations/utils/utils.hpp"

namespace InferenceEngine {

PreProcessInfo copyPreProcess(const PreProcessInfo& from) {
    PreProcessInfo to = from;
    if (from.getMeanVariant() == MEAN_IMAGE) {
        for (size_t i = 0; i < from.getNumberOfChannels(); i++) {
            auto& from_blob = from[i]->meanData;
            auto to_blob = make_blob_with_precision(from[i]->meanData->getTensorDesc());
            to_blob->allocate();
            ie_memcpy(to_blob->buffer(), to_blob->byteSize(), from_blob->cbuffer(), from_blob->byteSize());

            to.setMeanImageForChannel(to_blob, i);
        }
    }
    return to;
}

InputsDataMap copyInfo(const InputsDataMap& networkInputs) {
    InputsDataMap _networkInputs;
    for (const auto& it : networkInputs) {
        InputInfo::Ptr newPtr;
        if (it.second) {
            newPtr = std::make_shared<InputInfo>();
            newPtr->getPreProcess() = it.second->getPreProcess();
            newPtr->setInputData(std::make_shared<Data>(*it.second->getInputData()));
        }
        _networkInputs.emplace(it.first, newPtr);
    }
    return _networkInputs;
}

OutputsDataMap copyInfo(const OutputsDataMap& networkOutputs) {
    OutputsDataMap _networkOutputs;
    for (const auto& it : networkOutputs) {
        DataPtr newData;
        if (it.second) {
            newData = std::make_shared<Data>(*it.second);
        }
        _networkOutputs.emplace(it.first, newData);
    }
    return _networkOutputs;
}

IInferencePlugin::IInferencePlugin() : _executorManager(InferenceEngine::executorManager()), _isNewAPI(true) {}

void IInferencePlugin::VersionStore::copyFrom(const Version& v) {
    description = v.description;
    buildNumber = v.buildNumber;
    apiVersion = v.apiVersion;
}

IInferencePlugin::VersionStore::VersionStore(const Version& v) {
    copyFrom(v);
}

IInferencePlugin::VersionStore& IInferencePlugin::VersionStore::operator=(const VersionStore& v) {
    if (&v != this) {
        copyFrom(v);
    }
    return *this;
}

void IInferencePlugin::SetVersion(const Version& version) {
    _version = VersionStore(version);
}

const Version& IInferencePlugin::GetVersion() const {
    return _version;
}

std::string IInferencePlugin::GetName() const noexcept {
    return _pluginName;
}

void IInferencePlugin::SetName(const std::string& pluginName) noexcept {
    _pluginName = pluginName;
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::LoadNetwork(
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config) {
    return LoadNetwork(network, config, nullptr);
}

template <typename T>
std::map<std::string, std::shared_ptr<const T>> const_map_cast(const std::map<std::string, std::shared_ptr<T>>& map) {
    std::map<std::string, std::shared_ptr<const T>> res;
    for (auto&& v : map)
        res.emplace(v.first, std::const_pointer_cast<const T>(v.second));
    return res;
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::LoadNetwork(
    const CNNNetwork& orig_network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<RemoteContext>& context) {
    std::shared_ptr<IExecutableNetworkInternal> impl;

    // if IR `version` is not set, suppose it's IR v10 for old API
    // it allows to use operation names in set_ / get_tensor instead of tensor_names
    auto orig_function = orig_network.getFunction();
    std::shared_ptr<ov::Model> function;
    InferenceEngine::CNNNetwork network = orig_network;
    if (orig_function) {
        function = std::make_shared<ov::Model>(orig_function->get_results(),
                                               orig_function->get_sinks(),
                                               orig_function->get_parameters(),
                                               orig_function->get_variables(),
                                               orig_function->get_friendly_name());
        function->get_rt_info() = orig_function->get_rt_info();
    }
    if (function && !IsNewAPI()) {
        if (!function->has_rt_info("version")) {
            function->set_rt_info(int64_t(10), "version");

            // re-create `network` with new patched `function`
            using namespace InferenceEngine;
            OPENVINO_SUPPRESS_DEPRECATED_START
            const auto& orig_icnn = static_cast<const ICNNNetwork&>(orig_network);
            auto orig_impl =
                std::dynamic_pointer_cast<const details::CNNNetworkNGraphImpl>(orig_icnn.shared_from_this());
            OPENVINO_ASSERT(orig_impl != nullptr,
                            "Internal: orig_impl must be castable to details::CNNNetworkNGraphImpl");
            auto new_impl =
                std::make_shared<details::CNNNetworkNGraphImpl>(function, orig_impl->getExtensions(), IsNewAPI());
            network = CNNNetwork(new_impl);
            for (const auto& inputInfo : orig_network.getInputsInfo()) {
                auto toInfo = network.getInputsInfo().at(inputInfo.first);
                toInfo->setPrecision(inputInfo.second->getPrecision());
                toInfo->setLayout(inputInfo.second->getLayout());
                toInfo->getPreProcess() = inputInfo.second->getPreProcess();
            }
            for (const auto& outputInfo : orig_network.getOutputsInfo()) {
                auto toInfo = network.getOutputsInfo().at(outputInfo.first);
                toInfo->setPrecision(outputInfo.second->getPrecision());
                toInfo->setLayout(outputInfo.second->getLayout());
            }
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
    }

    if (nullptr == context) {
        impl = LoadExeNetworkImpl(network, config);
    } else {
        impl = LoadExeNetworkImpl(network, context, config);
    }

    SetExeNetworkInfo(impl, const_map_cast(network.getInputsInfo()), const_map_cast(network.getOutputsInfo()));
    if (function) {
        SetExeNetworkInfo(impl, function);
    }

    return impl;
}

ov::SoPtr<IExecutableNetworkInternal> IInferencePlugin::LoadNetwork(const std::string& modelPath,
                                                                    const std::map<std::string, std::string>& config) {
    auto cnnNet = GetCore()->ReadNetwork(modelPath, std::string());
    return GetCore()->LoadNetwork(cnnNet, GetName(), config);
}

void IInferencePlugin::AddExtension(const std::shared_ptr<IExtension>&) {
    IE_THROW(NotImplemented);
}

void IInferencePlugin::SetConfig(const std::map<std::string, std::string>&) {
    IE_THROW(NotImplemented);
}

void IInferencePlugin::SetProperties(const ov::AnyMap& config) {
    SetConfig(any_copy(config));
}

Parameter IInferencePlugin::GetConfig(const std::string&, const std::map<std::string, Parameter>&) const {
    IE_THROW(NotImplemented);
}

Parameter IInferencePlugin::GetMetric(const std::string&, const std::map<std::string, Parameter>&) const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<RemoteContext> IInferencePlugin::CreateContext(const ParamMap&) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<RemoteContext> IInferencePlugin::GetDefaultContext(const ParamMap&) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::ImportNetwork(
    const std::string& modelFileName,
    const std::map<std::string, std::string>& config) {
    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        IE_THROW(NetworkNotRead);
    }

    return ImportNetwork(blobFile, config);
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::ImportNetwork(
    std::istream& networkModel,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::ImportNetwork(
    std::istream& networkModel,
    const std::shared_ptr<RemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

void IInferencePlugin::SetCore(std::weak_ptr<ICore> core) {
    IE_ASSERT(!core.expired());
    _core = core;
    auto locked_core = _core.lock();
    if (locked_core)
        _isNewAPI = locked_core->isNewAPI();
}

std::shared_ptr<ICore> IInferencePlugin::GetCore() const noexcept {
    return _core.lock();
}

bool IInferencePlugin::IsNewAPI() const noexcept {
    return _isNewAPI;
}

const std::shared_ptr<ExecutorManager>& IInferencePlugin::executorManager() const {
    return _executorManager;
}

QueryNetworkResult IInferencePlugin::QueryNetwork(const CNNNetwork& network,
                                                  const std::map<std::string, std::string>& config) const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::LoadExeNetworkImpl(
    const CNNNetwork&,
    const std::map<std::string, std::string>&) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::LoadExeNetworkImpl(
    const CNNNetwork&,
    const std::shared_ptr<RemoteContext>&,
    const std::map<std::string, std::string>&) {
    IE_THROW(NotImplemented);
}

void IInferencePlugin::SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                                         const ConstInputsDataMap& inputs,
                                         const ConstOutputsDataMap& outputs) {
    IE_ASSERT(exeNetwork != nullptr);

    // Set inputs/outputs and pointer to plugin manually here
    exeNetwork->setNetworkInputs(copyInfo(constMapCast(inputs)));
    exeNetwork->setNetworkOutputs(copyInfo(constMapCast(outputs)));

    exeNetwork->SetPointerToPlugin(shared_from_this());
}

void IInferencePlugin::SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                                         const std::shared_ptr<const ov::Model>& function) {
    bool newAPI = IsNewAPI();
    InferenceEngine::SetExeNetworkInfo(exeNetwork, function, newAPI);
    exeNetwork->SetPointerToPlugin(shared_from_this());
}

std::unordered_set<std::string> GetRemovedNodes(const std::shared_ptr<const ov::Model>& originalFunction,
                                                const std::shared_ptr<const ov::Model>& transformedFunction) {
    std::unordered_set<std::string> result = {};
    std::unordered_set<std::string> transformedNodeNames = {};

    for (auto&& node : transformedFunction->get_ops()) {
        transformedNodeNames.emplace(node->get_friendly_name());
        for (auto&& fusedLayerName : ov::getFusedNamesVector(node))
            transformedNodeNames.emplace(fusedLayerName);
    }

    for (auto&& originalNode : originalFunction->get_ops()) {
        if (!InferenceEngine::details::contains(transformedNodeNames, originalNode->get_friendly_name()))
            result.emplace(originalNode->get_friendly_name());
    }

    return result;
}

std::unordered_set<std::string> GetSupportedNodes(
    const std::shared_ptr<const ov::Model>& model,
    std::function<void(std::shared_ptr<ov::Model>&)> transform,
    std::function<bool(const std::shared_ptr<ngraph::Node>)> is_node_supported) {
    return ov::get_supported_nodes(model, transform, is_node_supported);
}

void SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                       const std::shared_ptr<const ov::Model>& function,
                       bool new_api) {
    OPENVINO_ASSERT(exeNetwork != nullptr);
    OPENVINO_ASSERT(function != nullptr);

    std::vector<std::shared_ptr<const ov::Node>> const_params;
    std::vector<std::shared_ptr<const ov::Node>> const_results;

    std::unordered_set<std::string> leaf_names;
    bool add_operation_names = false;
    if (function->has_rt_info("version")) {
        const int64_t ir_version = function->get_rt_info<int64_t>("version");
        // here we decide whether we need to add operation_names as tensor names for
        // getInputs / getOutputs. Since these functions are designed to be used in new API only
        // always need to add operation names for IR v10
        add_operation_names = ir_version == 10;

        for (const auto& vals : {function->inputs(), function->outputs()}) {
            for (const auto& val : vals) {
                for (const auto& name : val.get_names()) {
                    leaf_names.insert(name);
                }
            }
        }
    }

    const auto& inputsInfo = exeNetwork->GetInputsInfo();
    const auto& outputsInfo = exeNetwork->GetOutputsInfo();
    OPENVINO_ASSERT(inputsInfo.size() == function->get_parameters().size());

    if (outputsInfo.size() != function->get_output_size()) {
        const auto& outputs = function->outputs();
        std::unordered_set<std::shared_ptr<ov::descriptor::Tensor>> output_tensors;
        std::transform(outputs.cbegin(),
                       outputs.cend(),
                       std::inserter(output_tensors, output_tensors.begin()),
                       [](const ov::Output<const ov::Node>& out) {
                           return out.get_tensor_ptr();
                       });

        OPENVINO_ASSERT(outputsInfo.size() == output_tensors.size(),
                        "outputsInfo.size() is: ",
                        outputsInfo.size(),
                        ", and function->get_output_size() is: ",
                        function->get_output_size(),
                        ". Number of duplicated outputs: ",
                        outputs.size() - output_tensors.size());
    }

    for (const auto& param : function->get_parameters()) {
        const auto& param_name = param->get_friendly_name();
        auto new_param = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
        new_param->set_friendly_name(param_name);
        if (add_operation_names) {
            OPENVINO_ASSERT(!new_api || leaf_names.find(param_name) == leaf_names.end() ||
                                param->output(0).get_names().find(param_name) != param->output(0).get_names().end(),
                            "Model operation names have collisions with tensor names.",
                            " Please use MO to generate new IR version, it should allow to avoid the issue");
            leaf_names.insert(param_name);
            new_param->output(0).get_tensor().add_names({param_name});
        }
        // WA: use CNNNetwork's precisions since plugins sometimes override their precisions
        // after transformation pipeline is run
        new_param->set_element_type(
            InferenceEngine::details::convertPrecision(inputsInfo.at(param_name)->getPrecision()));
        new_param->set_layout(param->get_layout());
        new_param->output(0).get_rt_info() = param->output(0).get_rt_info();
        new_param->validate_and_infer_types();
        const_params.emplace_back(new_param);
    }
    for (const auto& result : function->get_results()) {
        auto fake_param = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                  result->get_output_partial_shape(0));
        const std::string res_name = ov::op::util::create_ie_output_name(result->input_value(0));
        fake_param->set_friendly_name(res_name);
        fake_param->set_element_type(
            InferenceEngine::details::convertPrecision(outputsInfo.at(res_name)->getPrecision()));
        fake_param->validate_and_infer_types();
        auto new_result = result->copy_with_new_inputs({fake_param});
        new_result->set_friendly_name(result->get_friendly_name());
        if (add_operation_names) {
            OPENVINO_ASSERT(!new_api || leaf_names.find(res_name) == leaf_names.end() ||
                                result->output(0).get_names().find(res_name) != result->output(0).get_names().end(),
                            "Model operation names have collisions with tensor names.",
                            " Please use MO to generate new IR version, it should allow to avoid the issue");
            leaf_names.insert(res_name);
            new_result->output(0).get_tensor().add_names({res_name});
        }
        auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(new_result);
        OPENVINO_ASSERT(r, "Internal error. SetNetworkInfo failure casting output copy to Result");
        r->set_layout(result->get_layout());
        const_results.emplace_back(new_result);
    }

    exeNetwork->setInputs(const_params);
    exeNetwork->setOutputs(const_results);
}

std::shared_ptr<::ov::IPlugin> convert_plugin(const std::shared_ptr<InferenceEngine::IInferencePlugin>& from) {
    return ov::legacy_convert::convert_plugin(from);
}

}  //  namespace InferenceEngine
