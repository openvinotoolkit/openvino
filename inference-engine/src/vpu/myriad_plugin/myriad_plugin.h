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

class Engine : public ie::InferencePluginInternal {
public:
    explicit Engine(std::shared_ptr<IMvnc> mvnc);

    ~Engine() override {
        MyriadExecutor::closeDevices(_devicePool);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override;

    ie::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const ie::ICore* core,
            ie::ICNNNetwork& network,
            const std::map<std::string, std::string>& config) override;

    void QueryNetwork(
            const ie::ICNNNetwork& network,
            const std::map<std::string, std::string>& config,
            ie::QueryNetworkResult& res) const override;

    ie::IExecutableNetwork::Ptr ImportNetwork(
            const std::string& modelFileName,
            const std::map<std::string, std::string>& config) override;

    ie::Parameter GetConfig(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

    ie::Parameter GetMetric(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

    // Myriad plugin runs reshape internally so it needs reshapable network
    ie::ICNNNetwork& RemoveConstLayers(ie::ICNNNetwork& network) override {
        return network;
    }

private:
    MyriadConfig _parsedConfig;
    std::vector<DevicePtr> _devicePool;
    std::shared_ptr<IMvnc> _mvnc;
    std::shared_ptr<MyriadMetrics> _metrics;
};

}  // namespace MyriadPlugin
}  // namespace vpu
