// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/plugin_config.hpp"

namespace MultiDevicePlugin {

class LegacyAPIHelper {
public:
    static ov::AnyMap convert_legacy_properties(const std::map<std::string, std::string>& properties, bool is_new_api);
    static ov::AnyMap convert_legacy_properties(const ov::AnyMap& properties, bool is_new_api);
    static std::pair<std::string, ov::Any> convert_legacy_property(const std::pair<std::string, ov::Any>& legacy_property);
    static std::pair<std::string, ov::Any> convert_to_legacy_property(const std::pair<std::string, ov::Any>& property);
    static bool is_legacy_property(const std::pair<std::string, ov::Any>& property, bool is_new_api);
};

}  // namespace MultiDevicePlugin