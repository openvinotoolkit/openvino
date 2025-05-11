// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "config.hpp"

#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "properties.hpp"

using namespace ov::hetero;

Configuration::Configuration() {}

Configuration::Configuration(const ov::AnyMap& config, const Configuration& defaultCfg, bool throwOnUnsupported) {
    *this = defaultCfg;

    for (const auto& it : config) {
        const auto& key = it.first;
        const auto& value = it.second;

        if (ov::device::priorities == key) {
            device_priorities = value.as<std::string>();
        } else if (ov::hint::model_distribution_policy == key) {
            for (auto& row : value.as<std::set<ov::hint::ModelDistributionPolicy>>()) {
                if (row != ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL) {
                    OPENVINO_THROW(
                        "Wrong value ",
                        row,
                        " for property key ",
                        ov::hint::model_distribution_policy.name(),
                        ". HETERO plugin only support {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL}");
                }
            }
            modelDistributionPolicy = value.as<std::set<ov::hint::ModelDistributionPolicy>>();
        } else if (ov::cache_encryption_callbacks == key) {
            encryption_callbacks = value.as<EncryptionCallbacks>();
        } else {
            if (throwOnUnsupported)
                OPENVINO_THROW("Property was not found: ", key);
            device_properties.emplace(key, value);
        }
    }
}

ov::Any Configuration::get(const std::string& name) const {
    if (name == ov::device::priorities) {
        return {device_priorities};
    } else if (name == ov::hint::model_distribution_policy) {
        return {modelDistributionPolicy};
    } else {
        OPENVINO_THROW("Property was not found: ", name);
    }
}

std::vector<ov::PropertyName> Configuration::get_supported() const {
    static const std::vector<ov::PropertyName> names = {ov::device::priorities};
    return names;
}

ov::AnyMap Configuration::get_hetero_properties() const {
    return {{ov::device::priorities.name(), device_priorities},
            {ov::hint::model_distribution_policy.name(), modelDistributionPolicy}};
}

ov::AnyMap Configuration::get_device_properties() const {
    return device_properties;
}

bool Configuration::dump_dot_files() const {
    return std::getenv("OPENVINO_HETERO_VISUALIZE") != NULL;
}