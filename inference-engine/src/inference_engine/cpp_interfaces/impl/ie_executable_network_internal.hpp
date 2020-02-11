// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <ie_plugin_ptr.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"

namespace InferenceEngine {
/**
 * @brief minimum API to be implemented by plugin, which is used in ExecutableNetworkBase forwarding mechanism
 */
class ExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    typedef std::shared_ptr<ExecutableNetworkInternal> Ptr;

    virtual void setNetworkInputs(const InferenceEngine::InputsDataMap networkInputs) {
        _networkInputs = networkInputs;
    }

    virtual void setNetworkOutputs(const InferenceEngine::OutputsDataMap networkOutputs) {
        _networkOutputs = networkOutputs;
    }

    ConstOutputsDataMap GetOutputsInfo() const override {
        ConstOutputsDataMap outputMap;
        for (const auto& output : _networkOutputs) {
            outputMap[output.first] = output.second;
        }
        return outputMap;
    }

    ConstInputsDataMap GetInputsInfo() const override {
        ConstInputsDataMap inputMap;
        for (const auto& input : _networkInputs) {
            inputMap[input.first] = input.second;
        }
        return inputMap;
    }

    void Export(const std::string& /* modelFileName */) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    /**
     * @brief Writes IE header with magic and plugin name before the exported content
     * @param networkModel - output stream
     * @note
     */
    void Export(std::ostream& networkModel) override {
        std::stringstream strm;
        strm.write(exportMagic.data(), exportMagic.size());
        strm << _plugin->GetName() << std::endl;
        ExportImpl(strm);
        networkModel << strm.rdbuf();
    }

    /**
     * @brief Basic impl
     */
    virtual void ExportImpl(std::ostream& /* networkModel */) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>>& /* deployedTopology */) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr& /* graphPtr */) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetPointerToPluginInternal(IInferencePluginInternal::Ptr plugin) {
        _plugin = plugin;
    }

    std::vector<IMemoryStateInternal::Ptr> QueryState() override {
        // meaning base plugin reports as no state available - plugin owners need to create proper override of this
        return {};
    }

    void SetConfig(const std::map<std::string, Parameter>& config, ResponseDesc* /* resp */) override {
        if (config.empty()) {
            THROW_IE_EXCEPTION << "The list of configuration values is empty";
        }
        THROW_IE_EXCEPTION << "The following config value cannot be changed dynamically for ExecutableNetwork: "
                           << config.begin()->first;
    }

    void GetConfig(const std::string& /* name */, Parameter& /* result */, ResponseDesc* /* resp */) const override {
        THROW_IE_EXCEPTION << "GetConfig for executable network is not supported by this device";
    }

    void GetMetric(const std::string& /* name */, Parameter& /* result */, ResponseDesc* /* resp */) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetContext(RemoteContext::Ptr& /* pContext */, ResponseDesc* /* resp */) const override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

protected:
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;

    IInferencePluginInternal::Ptr _plugin;
};

}  // namespace InferenceEngine
