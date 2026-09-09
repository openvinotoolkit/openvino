// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "property_registration.hpp"

namespace intel_npu {

enum class ConfigMergeMode { Compile, Import, Query };

class PluginPropertyManager final : private PropertyRegistrationBase {
public:
    PluginPropertyManager(const std::shared_ptr<OptionsDesc>& options,
                          const ov::SoPtr<IEngineBackend>& backend,
                          const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper,
                          Logger& logger);

    PluginPropertyManager& operator=(const PluginPropertyManager& other) = delete;

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name, const ov::AnyMap& arguments = {}) const;
    bool isPropertySupported(const std::string& name, const ov::AnyMap& arguments = {}) const;

    std::pair<FilteredConfig, ov::AnyMap> getMergedConfigAndUnknownProperties(const ov::AnyMap& properties,
                                                                              ConfigMergeMode mergeMode);

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;

private:
    void registerProperties();
    std::optional<ov::intel_npu::CompilerType> resolveCompilerType(ov::intel_npu::CompilerType compilerType,
                                                                   const std::string& deviceId,
                                                                   const std::string& platform) const;
    void warnCompilerOnlyOptionSkipped(const std::string& key) const;

    FilteredConfig _config;

    ov::SoPtr<IEngineBackend> _backend;
    std::shared_ptr<CompilerOptionSupportHelper> _compilerOptionSupportHelper;
    Logger& _logger;

    mutable std::mutex _mutex;

    const std::vector<ov::PropertyName> _cachingProperties = [] {
        std::vector<ov::PropertyName> properties = {
            ov::cache_mode.name(),
            ov::enable_profiling.name(),
            ov::device::architecture.name(),
            ov::hint::execution_mode.name(),
            ov::hint::inference_precision.name(),
            ov::hint::performance_mode.name(),
            ov::intel_npu::batch_compiler_mode_settings.name(),
            ov::intel_npu::batch_mode.name(),
            ov::intel_npu::compilation_mode.name(),
            ov::intel_npu::compilation_mode_params.name(),
            ov::intel_npu::compiler_dynamic_quantization.name(),
            ov::intel_npu::compiler_type.name(),
            ov::intel_npu::dma_engines.name(),
            ov::intel_npu::driver_version.name(),
            ov::intel_npu::dynamic_shape_to_static.name(),
            ov::intel_npu::enable_strides_for.name(),
            ov::intel_npu::max_tiles.name(),
            ov::intel_npu::stepping.name(),
            ov::intel_npu::tiles.name(),
            ov::intel_npu::turbo.name(),
            ov::intel_npu::qdq_optimization.name(),
            ov::intel_npu::qdq_optimization_aggressive.name(),
        };
        for_each_cached_npuw_option([&](auto tag) {
            using Opt = typename decltype(tag)::type;
            properties.emplace_back(std::string{Opt::key()});
        });
        return properties;
    }();
};

}  // namespace intel_npu
