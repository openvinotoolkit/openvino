// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/configuration/plugin_configuration.hpp"
#include "myriad_config.h"

namespace vpu {

class MyriadConfiguration final : public PluginConfiguration, public MyriadPlugin::MyriadConfig {
public:
    MyriadConfiguration();

    // TODO: remove once all options are migrated
    void from(const std::map<std::string, std::string>& configuration);
    void fromAtRuntime(const std::map<std::string, std::string>& configuration);
};

}  // namespace vpu
