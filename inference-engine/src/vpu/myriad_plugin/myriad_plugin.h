// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "inference_engine.hpp"
#include "description_buffer.hpp"
#include "myriad_executable_network.h"
#include "myriad_mvnc_wraper.h"
#include "myriad_metrics.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>

namespace vpu {
namespace MyriadPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
public:
    explicit Engine(std::shared_ptr<IMvnc> mvnc);

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::ICore * core, InferenceEngine::ICNNNetwork &network,
                                                                       const std::map<std::string, std::string> &config) override;

    /**
     * @brief myriad plugin runs reshape internally so it needs reshapable network
     * @param network
     * @return
     */
    InferenceEngine::ICNNNetwork&  RemoveConstLayers(InferenceEngine::ICNNNetwork &network) override {
        return network;
    }

    void SetConfig(const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    /**
     * @deprecated Use the version with config parameter
     */
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network, InferenceEngine::QueryNetworkResult& res) const override;
    void QueryNetwork(const InferenceEngine::ICNNNetwork& network,
                      const std::map<std::string, std::string>& config, InferenceEngine::QueryNetworkResult& res) const override;

    InferenceEngine::IExecutableNetwork::Ptr ImportNetwork(const std::string &modelFileName, const std::map<std::string, std::string> &config) override;

    InferenceEngine::Parameter GetMetric(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    ~Engine() {
        MyriadExecutor::closeDevices(_devicePool);
    }

private:
    std::vector<DevicePtr> _devicePool;
    std::shared_ptr<IMvnc> _mvnc;
    std::shared_ptr<MyriadMetrics> _metrics;
};

}  // namespace MyriadPlugin
}  // namespace vpu
