// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "description_buffer.hpp"
#include "myriad_executable_network.h"
#include "myriad_mvnc_wrapper.h"
#include "myriad_metrics.h"
#include "vpu/configuration/plugin_configuration.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>

namespace vpu {
namespace MyriadPlugin {

class Engine : public ie::IInferencePlugin {
public:
    explicit Engine(std::shared_ptr<IMvnc> mvnc);

    void SetConfig(const std::map<std::string, std::string>& config) override;

    ie::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
            const ie::CNNNetwork& network,
            const std::map<std::string, std::string>& config) override;

    ie::QueryNetworkResult QueryNetwork(
            const ie::CNNNetwork& network,
            const std::map<std::string, std::string>& config) const override;

    using ie::IInferencePlugin::ImportNetwork;

    ie::IExecutableNetworkInternal::Ptr ImportNetwork(
            std::istream& model,
            const std::map<std::string, std::string>& config) override;

    ie::Parameter GetConfig(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

    ie::Parameter GetMetric(
            const std::string& name,
            const std::map<std::string, ie::Parameter>& options) const override;

private:
    PluginConfiguration _parsedConfig;
    std::shared_ptr<std::vector<DevicePtr>> _devicePool;
    std::shared_ptr<IMvnc> _mvnc;
    std::shared_ptr<MyriadMetrics> _metrics;

    class _EngineDevicePool {
        public:
            static std::shared_ptr<_EngineDevicePool> GetInstance(std::shared_ptr<IMvnc> mvnc);
            std::shared_ptr<std::vector<DevicePtr>> GetPool() {
                return _devicePool;
            }
            ~_EngineDevicePool();

        private:
            std::shared_ptr<std::vector<DevicePtr>> _devicePool;
            std::shared_ptr<IMvnc> _mvnc;
            _EngineDevicePool();
    };
};

}  // namespace MyriadPlugin
}  // namespace vpu
