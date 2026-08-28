// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <algorithm>
#include <memory>

#include "intel_npu/common/option_support_cache.hpp"

namespace intel_npu {

namespace {
OptionSupportCache::CacheKey toCacheKey(const ov::intel_npu::CompilerType compilerType) {
    return static_cast<OptionSupportCache::CacheKey>(compilerType);
}
}  // namespace

CompilerOptionSupportHelper::CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend,
                                                         const CompilerAdapterFactory& adapterFactory)
    : _backend(backend),
      _adapterFactory(adapterFactory),
      _optionSupportCache(std::make_shared<OptionSupportCache>()) {}

const std::shared_ptr<OptionSupportCache>& CompilerOptionSupportHelper::getOptionSupportCache() const {
    return _optionSupportCache;
}

bool CompilerOptionSupportHelper::isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                                    const std::string& optionName,
                                                    const std::optional<std::string>& optionValue) {
    const auto supportedCompilerTypes = _adapterFactory.getSupportedCompilerTypes();
    OPENVINO_ASSERT(std::find(supportedCompilerTypes.begin(), supportedCompilerTypes.end(), compilerType) !=
                        supportedCompilerTypes.end(),
                    "Unsupported compiler type");

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
    compiler = _adapterFactory.getCompiler(_backend, compilerType, "", _optionSupportCache);

    std::once_flag* optionsLoaded = nullptr;
    {
        std::lock_guard<std::mutex> lock(_supportedOptionsLoadedMutex);
        optionsLoaded = &_supportedOptionsLoaded[cacheKey];
    }
    std::call_once(*optionsLoaded, [&compiler]() {
        compiler->get_supported_options();
    });

    if (const auto cachedSupport = getCachedSupport(); cachedSupport.has_value()) {
        return cachedSupport.value();
    }

    return compiler->is_option_supported(optionName, optionValue);
}

}  // namespace intel_npu
