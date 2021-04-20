// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/parsed_config.hpp>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <memory>
#include <map>

#include <debug.h>
#include <ie_plugin_config.hpp>

#include <vpu/utils/string.hpp>

namespace vpu {

const std::unordered_set<std::string>& ParsedConfig::getCompileOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = merge(ParsedConfigBase::getCompileOptions(), {
        //
        // Private options
        //

        ie::MYRIAD_COPY_OPTIMIZATION,
    });
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfig::getRunTimeOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = ParsedConfigBase::getRunTimeOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

const std::unordered_set<std::string>& ParsedConfig::getDeprecatedOptions() const {
IE_SUPPRESS_DEPRECATED_START
    static const std::unordered_set<std::string> options = ParsedConfigBase::getDeprecatedOptions();
IE_SUPPRESS_DEPRECATED_END

    return options;
}

void ParsedConfig::parse(const std::map<std::string, std::string>& config) {
    ParsedConfigBase::parse(config);
}

}  // namespace vpu
