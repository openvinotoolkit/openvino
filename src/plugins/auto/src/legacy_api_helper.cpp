// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy_api_helper.hpp"
#include "ie_plugin_config.hpp"

namespace MultiDevicePlugin {

bool LegacyAPIHelper::is_legacy_property(const std::pair<std::string, ov::Any>& property, bool is_new_api) {
    static const std::vector<std::string> legacy_properties_list = {
    };

    static const std::vector<std::string> legacy_property_values_list = {
        InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
    };

    bool legacy_property = std::find(legacy_properties_list.begin(), legacy_properties_list.end(), property.first) != legacy_properties_list.end();
    bool need_value_conversion = !is_new_api &&
        std::find(legacy_property_values_list.begin(), legacy_property_values_list.end(), property.first) != legacy_property_values_list.end();

    return legacy_property || need_value_conversion;
}

ov::AnyMap LegacyAPIHelper::convert_legacy_properties(const std::map<std::string, std::string>& properties, bool is_new_api) {
    return convert_legacy_properties(ov::AnyMap(properties.begin(), properties.end()), is_new_api);
}

ov::AnyMap LegacyAPIHelper::convert_legacy_properties(const ov::AnyMap& properties, bool is_new_api) {
    ov::AnyMap converted_properties;
    for (auto& property : properties) {
        if (is_legacy_property(property, is_new_api)) {
            auto new_property = convert_legacy_property(property);
            converted_properties[new_property.first] = new_property.second;
        } else {
            converted_properties[property.first] = property.second;
        }
    }

    return converted_properties;
}

std::pair<std::string, ov::Any> LegacyAPIHelper::convert_legacy_property(const std::pair<std::string, ov::Any>& legacy_property) {
    auto legacy_name = legacy_property.first;
    if (legacy_name == InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY) {
        ov::Any converted_val{nullptr};
        auto legacy_val = legacy_property.second.as<std::string>();
        if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH) {
            converted_val = ov::hint::Priority::HIGH;
        } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED) {
            converted_val = ov::hint::Priority::MEDIUM;
        } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW) {
            converted_val = ov::hint::Priority::LOW;
        } else {
            converted_val = legacy_val;
        }

        return { ov::hint::model_priority.name(), converted_val };
    }
    OPENVINO_ASSERT(false, "[AUTO] Unhandled legacy property in convert_legacy_property method: ", legacy_property.first);
}

std::pair<std::string, ov::Any> LegacyAPIHelper::convert_to_legacy_property(const std::pair<std::string, ov::Any>& property) {
    auto name = property.first;
    if (name == ov::hint::model_priority.name()) {
        ov::Any legacy_val{nullptr};
        if (!property.second.empty()) {
            ov::hint::Priority val = property.second.as<ov::hint::Priority>();
            switch (val) {
            case ov::hint::Priority::LOW: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW; break;
            case ov::hint::Priority::MEDIUM: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED; break;
            case ov::hint::Priority::HIGH: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH; break;
            default: OPENVINO_ASSERT(false, "[GPU] Unsupported model priority value ", val);
            }
        }

        return { InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, legacy_val };
    }
    OPENVINO_ASSERT(false, "[AUTO] Unhandled legacy property in convert_to_legacy_property method: ", property.first);
}

}  // namespace MultiDevicePlugin
