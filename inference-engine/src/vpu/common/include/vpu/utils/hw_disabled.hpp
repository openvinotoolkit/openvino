// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/configuration/options/hw_black_list.hpp"

namespace vpu {

inline bool HwDisabled(const PluginConfiguration& configuration, const std::string& layerName) {
    auto hwBlackList = configuration.get<HwBlackListOption>();

    if (!hwBlackList.empty()) {
        return hwBlackList.count(layerName) != 0;
    }

    return false;
}

}  // namespace vpu
