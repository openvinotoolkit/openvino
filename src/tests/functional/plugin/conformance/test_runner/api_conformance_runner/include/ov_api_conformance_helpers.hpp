// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "api_conformance_helpers.hpp"

namespace ov {
namespace test {
namespace conformance {

inline const std::vector<ov::AnyMap> generate_ov_configs(const std::string& target_plugin,
                                                         const std::vector<ov::AnyMap>& config = {}) {
    std::pair<std::string, ov::Any> default_config;
    if (target_plugin ==  std::string(ov::test::utils::DEVICE_MULTI) ||
        target_plugin ==  std::string(ov::test::utils::DEVICE_AUTO) ||
        target_plugin ==  std::string(ov::test::utils::DEVICE_HETERO)) {
        default_config = ov::device::priorities(ov::test::conformance::targetDevice);
    } else if (target_plugin ==  std::string(ov::test::utils::DEVICE_BATCH)) {
        default_config = { CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(ov::test::conformance::targetDevice)};
    } else {
        throw std::runtime_error("Incorrect target device: " + target_plugin);
    }

    std::vector<ov::AnyMap> resultConfig;
    if (config.empty()) {
        return {{default_config}};
    }
    for (auto configItem : config) {
        configItem.insert(default_config);
        resultConfig.push_back(configItem);
    }
    return resultConfig;
}

}  // namespace conformance
}  // namespace test
}  // namespace ov
