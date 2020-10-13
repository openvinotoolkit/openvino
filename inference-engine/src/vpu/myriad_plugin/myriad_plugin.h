// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
        MyriadExecutor::closeDevices(_devicePool, _mvnc);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override;

    ie::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const ie::ICNNNetwork& network,
            const std::map<std::string, std::string>& config) override;

    ie::QueryNetworkResult QueryNetwork(
            const ie::ICNNNetwork& network,
            const std::map<std::string, std::string>& config) const override;

    using ie::InferencePluginInternal::ImportNetwork;

    ie::ExecutableNetwork ImportNetwork(
            const std::string& modelFileName,
            const std::map<std::string, std::string>& config) override;

    ie::ExecutableNetwork ImportNetwork(
            std::istream& model,
            const std::map<std::string, std::string>& config) override;

    ie::Parameter GetConfig(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

    ie::Parameter GetMetric(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

private:
    MyriadConfig _parsedConfig;
    std::vector<DevicePtr> _devicePool;
    std::shared_ptr<IMvnc> _mvnc;
    std::shared_ptr<MyriadMetrics> _metrics;
};

}  // namespace MyriadPlugin
}  // namespace vpu
