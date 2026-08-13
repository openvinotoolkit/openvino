// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <memory>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/option_support_cache.hpp"

namespace intel_npu {

namespace {
OptionSupportCache::CacheKey toCacheKey(const ov::intel_npu::CompilerType compilerType) {
    return static_cast<OptionSupportCache::CacheKey>(compilerType);
}
}  // namespace

CompilerOptionSupportHelper::CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend)
    : _backend(backend),
      _optionSupportCache(std::make_shared<OptionSupportCache>()) {}

const std::shared_ptr<OptionSupportCache>& CompilerOptionSupportHelper::getOptionSupportCache() const {
    return _optionSupportCache;
}

bool CompilerOptionSupportHelper::isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                                    const std::string& optionName,
                                                    const std::optional<std::string>& optionValue) {
    OPENVINO_ASSERT(compilerType != ov::intel_npu::CompilerType::PREFER_PLUGIN,
                    "Expected concrete compiler type before cache lookup");
    const auto cacheKey = toCacheKey(compilerType);

    const auto getCachedSupport = [&]() -> std::optional<bool> {
        if (optionValue.has_value()) {
            return std::nullopt;
        }
        return _optionSupportCache->isOptionSupported(cacheKey, optionName);
    };

    if (const auto cachedSupport = getCachedSupport(); cachedSupport.has_value()) {
        return cachedSupport.value();
    }

    std::unique_ptr<ICompilerAdapter> compiler;
    try {
        compiler = CompilerAdapterFactory().getCompiler(_backend, compilerType, "", _optionSupportCache);
    } catch (...) {
        return false;
    }

    std::once_flag& optionsLoaded = (compilerType == ov::intel_npu::CompilerType::DRIVER)
                                        ? _driverSupportedOptionsLoaded
                                        : _pluginSupportedOptionsLoaded;
    std::call_once(optionsLoaded, [&compiler]() {
        compiler->get_supported_options();
    });

    if (const auto cachedSupport = getCachedSupport(); cachedSupport.has_value()) {
        return cachedSupport.value();
    }

    return compiler->is_option_supported(optionName, optionValue);
}

}  // namespace intel_npu
