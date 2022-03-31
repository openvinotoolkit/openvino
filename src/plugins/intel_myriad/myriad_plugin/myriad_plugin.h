// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "description_buffer.hpp"
#include "myriad_executable_network.h"
#include "myriad_metrics.h"
#include "myriad_mvnc_wrapper.h"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {
namespace MyriadPlugin {

class Engine : public ie::IInferencePlugin {
public:
    explicit Engine(std::shared_ptr<IMvnc> mvnc);

    void SetConfig(const std::map<std::string, std::string>& config) override;

    ie::IExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const ie::CNNNetwork& network,
                                                           const std::map<std::string, std::string>& config) override;

    ie::QueryNetworkResult QueryNetwork(const ie::CNNNetwork& network,
                                        const std::map<std::string, std::string>& config) const override;

    using ie::IInferencePlugin::ImportNetwork;

    ie::IExecutableNetworkInternal::Ptr ImportNetwork(std::istream& model,
                                                      const std::map<std::string, std::string>& config) override;

    ie::Parameter GetConfig(const std::string& name,
                            const std::map<std::string, ie::Parameter>& options) const override;

    ie::Parameter GetMetric(const std::string& name,
                            const std::map<std::string, ie::Parameter>& options) const override;


private:
    static std::shared_ptr<DevicesManager> _devicesManager;
    PluginConfiguration _parsedConfig;
    std::shared_ptr<MyriadMetrics> _metrics;
};

}  // namespace MyriadPlugin
}  // namespace vpu
