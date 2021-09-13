// Copyright (C) 2018-2021 Intel Corporation
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
#include <string>

#include "blob_factory.hpp"
#include "exec_graph_info.hpp"
#include "ie_icore.hpp"
#include "ie_iextension.h"
#include "ie_input_info.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_parameter.hpp"

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

void IInferencePlugin::VersionStore::copyFrom(const Version& v) {
    _dsc = v.description;
    _buildNumber = v.buildNumber;
    description = _dsc.c_str();
    buildNumber = _buildNumber.c_str();
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
    const CNNNetwork& network,
    const std::map<std::string, std::string>& config,
    const std::shared_ptr<IRemoteContext>& context) {
    std::shared_ptr<IExecutableNetworkInternal> impl;
    if (nullptr == context) {
        impl = LoadExeNetworkImpl(network, config);
    } else {
        impl = LoadExeNetworkImpl(network, context, config);
    }

    SetExeNetworkInfo(impl, const_map_cast(network.getInputsInfo()), const_map_cast(network.getOutputsInfo()));
    auto function = network.getFunction();
    if (function) {
        SetExeNetworkInfo(impl, std::const_pointer_cast<ov::Function>(function));
    }

    return impl;
}

std::shared_ptr<IExecutableNetworkInternal> IInferencePlugin::LoadNetwork(
    const std::string& modelPath,
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

Parameter IInferencePlugin::GetConfig(const std::string&, const std::map<std::string, Parameter>&) const {
    IE_THROW(NotImplemented);
}

Parameter IInferencePlugin::GetMetric(const std::string&, const std::map<std::string, Parameter>&) const {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IRemoteContext> IInferencePlugin::CreateContext(const ParamMap&) {
    IE_THROW(NotImplemented);
}

std::shared_ptr<IRemoteContext> IInferencePlugin::GetDefaultContext(const ParamMap&) {
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
    const std::shared_ptr<IRemoteContext>& context,
    const std::map<std::string, std::string>& config) {
    IE_THROW(NotImplemented);
}

void IInferencePlugin::SetCore(std::weak_ptr<ICore> core) {
    IE_ASSERT(!core.expired());
    _core = core;
}

std::shared_ptr<ICore> IInferencePlugin::GetCore() const noexcept {
    return _core.lock();
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
    const std::shared_ptr<IRemoteContext>&,
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

    ngraph::ParameterVector parameters;
    ngraph::ResultVector results;
    std::vector<ngraph::Output<ngraph::Node>> node_outputs;

    for (auto&& input : inputs) {
        auto tensor_desc = input.second->getTensorDesc();
        auto dims = tensor_desc.getDims();
        parameters.push_back(
            std::make_shared<ngraph::op::v0::Parameter>(details::convertPrecision(tensor_desc.getPrecision()),
                                                        std::vector<ov::Dimension>{dims.begin(), dims.end()}));
        parameters.back()->set_friendly_name(input.first);
        node_outputs.push_back(parameters.back()->output(0));
    }

    auto node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(node_outputs, outputs.size());

    int i = 0;
    for (auto&& output : outputs) {
        auto tensor_desc = output.second->getTensorDesc();
        auto dims = tensor_desc.getDims();
        node->set_output_type(i,
                              details::convertPrecision(tensor_desc.getPrecision()),
                              std::vector<ov::Dimension>{dims.begin(), dims.end()});
        results.push_back(std::make_shared<ngraph::op::v0::Result>(node->output(i)));
        ++i;
    }
    exeNetwork->setRuntimeFunction(std::make_shared<ov::Function>(results, parameters, "execution_info"));
    exeNetwork->SetPointerToPlugin(shared_from_this());
}

void IInferencePlugin::SetExeNetworkInfo(const std::shared_ptr<IExecutableNetworkInternal>& exeNetwork,
                                         const std::shared_ptr<ov::Function>& function) {
    IE_ASSERT(exeNetwork != nullptr);
    IE_ASSERT(function != nullptr);

    ngraph::ParameterVector parameters;
    ngraph::ResultVector results;
    ngraph::NodeVector nodes;

    std::map<ngraph::Output<ngraph::Node>, ngraph::Output<ngraph::Node>> output_map;

    for (auto&& node : function->get_ordered_ops()) {
        ngraph::Node* new_node = nullptr;
        if (ngraph::is_type<ngraph::op::Parameter>(node)) {
            parameters.push_back(std::static_pointer_cast<ngraph::op::v0::Parameter>(node->clone_with_new_inputs({})));
            for (std::size_t i = 0; i < node->outputs().size(); ++i) {
                output_map.emplace(node->output(i), parameters.back()->output(i));
            }
            new_node = parameters.back().get();
        } else {
            std::vector<ngraph::Output<ngraph::Node>> outputs;
            for (auto&& input : node->inputs()) {
                outputs.emplace_back(output_map.at(input.get_source_output()));
            }
            if (ngraph::is_type<ngraph::op::Result>(node)) {
                results.push_back(
                    std::static_pointer_cast<ngraph::op::v0::Result>(node->clone_with_new_inputs(outputs)));
                new_node = results.back().get();
            } else {
                nodes.push_back(
                    std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(outputs, node->outputs().size()));
                new_node = nodes.back().get();
                for (std::size_t i = 0; i < node->outputs().size(); ++i) {
                    auto output = node->output(i);
                    output_map.emplace(output, nodes.back()->output(i));
                    new_node->set_output_type(i, output.get_element_type(), output.get_partial_shape());
                }
            }
        }
        IE_ASSERT(new_node != nullptr);
        new_node->set_friendly_name(node->get_friendly_name());
        new_node->get_rt_info()[ExecGraphInfoSerialization::PERF_COUNTER] =
            std::make_shared<::ngraph::VariantWrapper<std::string>>("not_executed");
        new_node->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] =
            std::make_shared<::ngraph::VariantWrapper<std::string>>(node->get_friendly_name());
    }
    exeNetwork->setRuntimeFunction(
        std::make_shared<ov::Function>(results, parameters, function->get_friendly_name() + "_execution_info"));
    exeNetwork->SetPointerToPlugin(shared_from_this());
}

}  //  namespace InferenceEngine
