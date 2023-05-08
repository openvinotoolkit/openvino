// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include "proxy_plugin.hpp"

namespace {

std::vector<std::string> split(const std::string& str, const std::string& delim = ",") {
    std::vector<std::string> result;
    std::string::size_type start(0);
    std::string::size_type end = str.find(delim);
    while (end != std::string::npos) {
        result.emplace_back(str.substr(start, end - start));
        start = end + delim.size();
        end = str.find(delim, start);
    }
    result.emplace_back(str.substr(start, end - start));
    return result;
}

}  // namespace

std::string ov::proxy::restore_order(const std::string& original_order) {
    std::string result;
    std::vector<std::string> dev_order;
    auto fallback_properties = split(original_order);
    if (fallback_properties.size() == 1) {
        // Simple case I shouldn't restore the right order
        dev_order = split(fallback_properties.at(0), "->");
    } else {
        OPENVINO_THROW("Cannot restore fallback devices priority from the next config: ", original_order);
    }
    for (const auto& dev : dev_order) {
        if (!result.empty())
            result += " ";
        result += dev;
    }
    return result;
}

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_proxy_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::proxy::Plugin, version)
