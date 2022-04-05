// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "any_copy.hpp"

#include <sstream>

#include "ie_plugin_config.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}
}  // namespace ov
