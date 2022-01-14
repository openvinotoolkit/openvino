// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/none_layers.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"
#include "vpu/utils/error.hpp"

#include <vpu/utils/string.hpp>

namespace vpu {

void NoneLayersOption::validate(const std::string& value) {
    try {
        splitStringList<NoneLayersOption::value_type>(value, ',');
    } catch(const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}")", key(), value);
    }
}

void NoneLayersOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string NoneLayersOption::key() {
    return InferenceEngine::MYRIAD_NONE_LAYERS;
}

details::Access NoneLayersOption::access() {
    return details::Access::Private;
}

details::Category NoneLayersOption::category() {
    return details::Category::CompileTime;
}

std::string NoneLayersOption::defaultValue() {
    return std::string();
}

NoneLayersOption::value_type NoneLayersOption::parse(const std::string& value) {
    NoneLayersOption::value_type splitList;
    try {
        splitList = splitStringList<NoneLayersOption::value_type>(value, ',');
    } catch(const std::exception& e) {
        VPU_THROW_FORMAT(R"(unexpected {} option value "{}")", key(), value);
    }
    return splitList;
}

}  // namespace vpu
