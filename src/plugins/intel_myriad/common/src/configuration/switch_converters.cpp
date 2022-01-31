// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/containers.hpp"
#include "vpu/configuration/switch_converters.hpp"

#include "ie_plugin_config.hpp"

namespace vpu {

const std::unordered_map<std::string, bool>& string2switch() {
    static const std::unordered_map<std::string, bool> converters = {
        {CONFIG_VALUE(NO), false},
        {CONFIG_VALUE(YES), true}
    };
    return converters;
}

const std::unordered_map<bool, std::string>& switch2string() {
    static const auto converters = inverse(string2switch());
    return converters;
}

}  // namespace vpu
