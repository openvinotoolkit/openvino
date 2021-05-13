// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include "cpp_interfaces/impl/ie_executable_network_internal.hpp"
#include "cpp_interfaces/impl/ie_infer_request_2_internal.hpp"
#include "cpp/ie_executable_network.hpp"

namespace InferenceEngine {

class ExecutableNetwork2Internal : public IExecutableNetworkInternal {
    ExecutableNetwork actual;
public:
    explicit ExecutableNetwork2Internal(const ExecutableNetwork & exec) : actual(exec) {
        if (!actual) {
            IE_THROW(NotAllocated);
        }
    }

    virtual void setNetworkInputs(const InferenceEngine::InputsDataMap networkInputs) {
        // should call internal methods
        IE_THROW(NotImplemented);
    }

    virtual void setNetworkOutputs(const InferenceEngine::OutputsDataMap networkOutputs) {
        // should call internal methods
        IE_THROW(NotImplemented);
    }

    ConstOutputsDataMap GetOutputsInfo() const override {
        return actual.GetOutputsInfo();
    }

    ConstInputsDataMap GetInputsInfo() const override {
        return actual.GetInputsInfo();
    }

    void Export(const std::string& modelFileName) override {
        actual.Export(modelFileName);
    }

    void Export(std::ostream& networkModel) override {
        actual.Export(networkModel);
    }

    CNNNetwork GetExecGraphInfo() override {
        return actual.GetExecGraphInfo();
    }

    std::vector<IVariableStateInternal::Ptr> QueryState() override {
        IE_SUPPRESS_DEPRECATED_START
        std::vector<IVariableStateInternal::Ptr> states;
        for (auto & state : actual.QueryState()) {
            states.push_back(std::make_shared<VariableState2Internal>(state));
        }
        return states;
        IE_SUPPRESS_DEPRECATED_END
    }

    void SetConfig(const std::map<std::string, Parameter>& config) override {
        actual.SetConfig(config);
    }

    Parameter GetConfig(const std::string& name) const override {
        return actual.GetConfig(name);
    }

    Parameter GetMetric(const std::string& name) const override {
        return actual.GetConfig(name);
    }

    RemoteContext::Ptr GetContext() const override {
        return actual.GetContext();
    }

    IInferRequestInternal::Ptr CreateInferRequest() override {
        return std::make_shared<InferRequest2Internal>(actual.CreateInferRequest());
    }
};

}  // namespace InferenceEngine
