// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/private_plugin_config.hpp"
#include "vpu/utils/containers.hpp"
#include "vpu/configuration/options/enable_async_dma.hpp"
#include "vpu/configuration/switch_converters.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

namespace vpu {

void EnableAsyncDMAOption::validate(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
}

void EnableAsyncDMAOption::validate(const PluginConfiguration& configuration) {
    validate(configuration[key()]);
}

std::string EnableAsyncDMAOption::key() {
    return InferenceEngine::MYRIAD_ENABLE_ASYNC_DMA;
}

details::Access EnableAsyncDMAOption::access() {
    return details::Access::Private;
}

details::Category EnableAsyncDMAOption::category() {
    return details::Category::CompileTime;
}

std::string EnableAsyncDMAOption::defaultValue() {
    return InferenceEngine::PluginConfigParams::YES;
}

EnableAsyncDMAOption::value_type EnableAsyncDMAOption::parse(const std::string& value) {
    const auto& converters = string2switch();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(converters.count(value) != 0, R"(unexpected {} option value "{}", only {} are supported)",
        key(), value, getKeys(converters));
    return converters.at(value);
}

}  // namespace vpu
