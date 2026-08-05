// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_options_cache.hpp"

#include <algorithm>
#include <memory>

#include "intel_npu/common/compiler_adapter_factory.hpp"

namespace intel_npu {

std::mutex CompilerOptionsCache::_mutex;
CompilerOptionsCache::CompilerTypeOptionsState CompilerOptionsCache::_driverCompilerOptionsState{};
CompilerOptionsCache::CompilerTypeOptionsState CompilerOptionsCache::_pluginCompilerOptionsState{};

bool CompilerOptionsCache::isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                             const std::string& optionName,
                                             const std::optional<std::string>& optionValue,
                                             const ov::SoPtr<IEngineBackend>& engineBackend,
                                             const std::optional<uint32_t>& compilerSupportVersion) {
    std::lock_guard<std::mutex> lock(_mutex);

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    const auto getCompiler = [&]() -> ICompilerAdapter* {
        if (compiler == nullptr) {
            compiler = CompilerAdapterFactory().getCompiler(engineBackend, compilerType, "");
        }
        return compiler.get();
    };

    auto& compilerOptionsState = getStateForCompilerType(compilerType);
    const auto optionCacheKey = buildOptionCacheKey(optionName, optionValue);

    if (compilerOptionsState.supportedOptionsQueried && compilerOptionsState.supportedOptions.has_value()) {
        const bool isCachedAsSupported =
            compilerOptionsState.legacy
                ? isLegacyOptionCached(compilerOptionsState, optionCacheKey, compilerSupportVersion)
                : isOptionCached(compilerOptionsState, optionCacheKey);
        if (isCachedAsSupported) {
            return true;
        }
    }

    if (!compilerOptionsState.supportedOptionsQueried) {
        auto* compilerPtr = getCompiler();
        OPENVINO_ASSERT(compilerPtr != nullptr, "Compiler must be present to filter properties by compiler support");

        compilerOptionsState.supportedOptions = compilerPtr->get_supported_options();
        compilerOptionsState.supportedOptionsQueried = true;
        if (!compilerOptionsState.supportedOptions.has_value()) {
            compilerOptionsState.legacy = true;
            compilerOptionsState.compilerVersion = compilerPtr->get_version();
        } else {
            if (isOptionCached(compilerOptionsState, optionCacheKey)) {
                return true;
            }
        }
    }

    if (compilerOptionsState.legacy) {
        if (compilerSupportVersion.has_value() &&
            compilerOptionsState.compilerVersion >= compilerSupportVersion.value()) {
            addSupportedOptionImpl(compilerOptionsState,
                                   buildLegacyOptionCacheKey(optionCacheKey, compilerSupportVersion.value()));
            return true;
        }
        return false;
    }

    auto* compilerPtr = getCompiler();
    if (compilerPtr == nullptr) {
        return false;
    }

    const bool supported = compilerPtr->is_option_supported(optionName, optionValue);
    if (supported) {
        addSupportedOptionImpl(compilerOptionsState, optionCacheKey);
    }

    return supported;
}

CompilerOptionsCache::CompilerTypeOptionsState& CompilerOptionsCache::getStateForCompilerType(
    ov::intel_npu::CompilerType compilerType) {
    if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        return _driverCompilerOptionsState;
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        return _pluginCompilerOptionsState;
    }

    OPENVINO_THROW("Unsupported compiler type in CompilerOptionsCache. Expected DRIVER or PLUGIN.");
}

std::string CompilerOptionsCache::buildOptionCacheKey(const std::string& optionName,
                                                      const std::optional<std::string>& optionValue) {
    if (!optionValue.has_value()) {
        return optionName;
    }

    std::string key = optionName;
    key += '=';
    key += optionValue.value();
    return key;
}

std::string CompilerOptionsCache::buildLegacyOptionCacheKey(const std::string& optionCacheKey,
                                                            uint32_t compilerSupportVersion) {
    std::string key = optionCacheKey;
    key += "@v=";
    key += std::to_string(compilerSupportVersion);
    return key;
}

bool CompilerOptionsCache::isOptionCached(const CompilerTypeOptionsState& compilerOptionsState,
                                          const std::string& optionCacheKey) {
    if (!compilerOptionsState.supportedOptions.has_value()) {
        return false;
    }

    const auto& supportedOptions = compilerOptionsState.supportedOptions.value();
    return std::find(supportedOptions.begin(), supportedOptions.end(), optionCacheKey) != supportedOptions.end();
}

bool CompilerOptionsCache::isLegacyOptionCached(const CompilerTypeOptionsState& compilerOptionsState,
                                                const std::string& optionCacheKey,
                                                const std::optional<uint32_t>& compilerSupportVersion) {
    if (!compilerOptionsState.supportedOptions.has_value()) {
        return false;
    }

    const auto& supportedOptions = compilerOptionsState.supportedOptions.value();
    if (compilerSupportVersion.has_value()) {
        const auto cacheKey = buildLegacyOptionCacheKey(optionCacheKey, compilerSupportVersion.value());
        return std::find(supportedOptions.begin(), supportedOptions.end(), cacheKey) != supportedOptions.end();
    }

    const std::string prefix = optionCacheKey + "@v=";
    return std::any_of(supportedOptions.begin(), supportedOptions.end(), [&](const std::string& value) {
        if (value == optionCacheKey) {
            return true;
        }
        return value.size() > prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    });
}

void CompilerOptionsCache::addSupportedOptionImpl(CompilerTypeOptionsState& compilerOptionsState,
                                                  const std::string& optionName) {
    if (compilerOptionsState.supportedOptions.has_value()) {
        const auto& supportedOptions = compilerOptionsState.supportedOptions.value();
        if (std::find(supportedOptions.begin(), supportedOptions.end(), optionName) != supportedOptions.end()) {
            return;
        }
    }

    if (!compilerOptionsState.supportedOptionsQueried) {
        compilerOptionsState.supportedOptionsQueried = true;
        compilerOptionsState.supportedOptions = std::vector<std::string>{};
    }

    if (!compilerOptionsState.supportedOptions.has_value()) {
        compilerOptionsState.supportedOptions = std::vector<std::string>{};
    }

    compilerOptionsState.supportedOptions->push_back(optionName);
}

}  // namespace intel_npu
