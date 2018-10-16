// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <ie_plugin_ptr.hpp>
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_async_request_internal.hpp"

namespace InferenceEngine {

class InferencePluginInternal;

typedef std::shared_ptr<InferencePluginInternal> InferencePluginInternalPtr;

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
        for (const auto & output : _networkOutputs) {
            outputMap[output.first] = output.second;
        }
        return outputMap;
    }

    ConstInputsDataMap GetInputsInfo() const override {
        ConstInputsDataMap  inputMap;
        for (const auto & input : _networkInputs) {
            inputMap[input.first] = input.second;
        }
        return inputMap;
    }

    void Export(const std::string &modelFileName) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void GetMappedTopology(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }

    void SetPointerToPluginInternal(InferencePluginInternalPtr plugin) {
        _plugin = plugin;
    }

    std::vector<IMemoryStateInternal::Ptr>  QueryState() override {
        // meaning base plugin reports as no state available - plugin owners need to create proper override of this
        return {};
    }


protected:
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;

    InferencePluginInternalPtr _plugin;
};

}  // namespace InferenceEngine
