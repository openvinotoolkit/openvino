// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "api_conformance_helpers.hpp"

namespace ov {
namespace test {
namespace conformance {

inline const std::vector<ov::AnyMap> generate_ov_configs(const std::vector<ov::AnyMap>& config = {}) {
    ov::AnyMap default_config;
    if (ov::test::conformance::targetDevice ==  std::string(ov::test::utils::DEVICE_MULTI) ||
        ov::test::conformance::targetDevice ==  std::string(ov::test::utils::DEVICE_AUTO) ||
        ov::test::conformance::targetDevice ==  std::string(ov::test::utils::DEVICE_HETERO)) {
        default_config = {ov::device::priorities(ov::test::conformance::targetDevice)};
    } else if (ov::test::conformance::targetDevice == std::string(ov::test::utils::DEVICE_BATCH)) {
        default_config = {{ CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(ov::test::conformance::targetDevice)}};
    } else {
        default_config = {};
    }

    std::vector<ov::AnyMap> resultConfig;
    if (config.empty()) {
        return { default_config };
    }
    for (auto configItem : config) {
        configItem.insert(default_config.begin(), default_config.end());
        resultConfig.push_back(configItem);
    }
    return resultConfig;
}

inline const std::vector<std::string> return_device_combination() {
    std::vector<std::string> sw_plugins{ov::test::utils::DEVICE_HETERO, ov::test::utils::DEVICE_AUTO,
                                        ov::test::utils::DEVICE_BATCH, ov::test::utils::DEVICE_MULTI};
    if (std::find(sw_plugins.begin(), sw_plugins.end(), ov::test::conformance::targetDevice) != sw_plugins.end()) {
        std::string sw_device = generate_complex_device_name(ov::test::utils::DEVICE_TEMPLATE);
        return {sw_device};
    }

    return {ov::test::conformance::targetDevice};
}

inline std::vector<std::pair<std::string, std::string>> generate_ov_pairs_plugin_name_by_device() {
    std::vector<std::pair<std::string, std::string>> res;
    for (const auto& device : return_device_combination()) {
        std::string real_device = device.substr(0, device.find(':'));
        res.push_back(std::make_pair(get_plugin_lib_name_by_device(real_device),
                                     real_device));
    }
    return res;
}


}  // namespace conformance
}  // namespace test
}  // namespace ov
