// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/configuration/options/dump_internal_graph_file_name.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void DumpInternalGraphFileNameOption::validate(const std::string& value) {}

void DumpInternalGraphFileNameOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string DumpInternalGraphFileNameOption::key() {
    return InferenceEngine::MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME;
}

details::Access DumpInternalGraphFileNameOption::access() {
    return details::Access::Private;
}

details::Category DumpInternalGraphFileNameOption::category() {
    return details::Category::CompileTime;
}

std::string DumpInternalGraphFileNameOption::defaultValue() {
    return std::string();
}

DumpInternalGraphFileNameOption::value_type DumpInternalGraphFileNameOption::parse(const std::string& value) {
    return value;
}

}  // namespace vpu
