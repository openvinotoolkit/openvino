// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_itask_executor.hpp>

namespace AutoPlugin {

using DeviceName = std::string;

typedef std::promise<InferenceEngine::SoExecutableNetworkInternal> NetworkPromise;
typedef std::future<InferenceEngine::SoExecutableNetworkInternal> NetworkFuture;
typedef std::shared_future<InferenceEngine::SoExecutableNetworkInternal> NetworkSharedFuture;
typedef std::shared_ptr<NetworkPromise> NetworkPromiseSharedPtr;

class AutoExecutableNetwork : public InferenceEngine::IExecutableNetworkInternal {
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    explicit AutoExecutableNetwork(AutoPlugin::NetworkPromiseSharedPtr networkFirstReady,
                                   AutoPlugin::NetworkPromiseSharedPtr networkActualNeeded);

    void Export(std::ostream& networkModel) override;
    InferenceEngine::RemoteContext::Ptr GetContext() const override;
    InferenceEngine::CNNNetwork GetExecGraphInfo() override;
    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name) const override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;

    ~AutoExecutableNetwork();

private:
    InferenceEngine::SoExecutableNetworkInternal _networkFirstReady;

    InferenceEngine::SoExecutableNetworkInternal _networkActualNeeded;
    AutoPlugin::NetworkPromiseSharedPtr _networkPromiseActualNeeded;
    AutoPlugin::NetworkFuture futureActualNetwork; // for requests
    void wait_for_actual_device() const {
        // _networkActualNeeded = _networkPromiseActualNeeded->get_future().get();
        // todo : catch the st std::future_error / std::future_errc::promise_already_satisfied
        // todo: make the two members above volatile to keep this method const
    }
};

}  // namespace AutoPlugin
