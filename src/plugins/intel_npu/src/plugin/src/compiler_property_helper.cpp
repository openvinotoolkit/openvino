// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_property_helper.hpp"

#include <memory>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/compiler_options_cache.hpp"

namespace intel_npu {

bool isCompilerOptionSupported(ov::intel_npu::CompilerType compilerType,
                               const std::string& optionName,
                               const std::optional<std::string>& optionValue,
                               const ov::SoPtr<IEngineBackend>& engineBackend,
                               const std::optional<uint32_t>& compilerSupportVersion) {
    std::unique_ptr<ICompilerAdapter> compiler;
    const auto getCompiler = [&]() -> ICompilerAdapter* {
        if (compiler == nullptr) {
            try {
                compiler = CompilerAdapterFactory().getCompiler(engineBackend, compilerType, "");
            } catch (...) {
                return nullptr;
            }
        }
        return compiler.get();
    };

    // Resolve PREFER_PLUGIN to a concrete compiler type before cache lookup.
    if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        auto* compilerPtr = getCompiler();
        if (compilerPtr == nullptr) {
            return false;
        }
        return compilerPtr->is_option_supported(optionName, optionValue);
    }

    if (CompilerOptionsCache::isOptionSupported(compilerType, optionName, optionValue, compilerSupportVersion)) {
        return true;
    }

    auto* compilerPtr = getCompiler();
    if (compilerPtr == nullptr) {
        return false;
    }

    return compilerPtr->is_option_supported(optionName, optionValue);
}

std::optional<std::vector<std::string>> getCompilerSupportedOptions(ov::intel_npu::CompilerType compilerType,
                                                                    const ov::SoPtr<IEngineBackend>& engineBackend) {
    auto effectiveCompilerType = compilerType;
    std::unique_ptr<ICompilerAdapter> compiler;
    const auto getCompiler = [&]() -> ICompilerAdapter* {
        if (compiler == nullptr) {
            try {
                compiler = CompilerAdapterFactory().getCompiler(engineBackend, effectiveCompilerType, "");
            } catch (...) {
                return nullptr;
            }
        }
        return compiler.get();
    };

    if (effectiveCompilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        auto* compilerPtr = getCompiler();
        if (compilerPtr == nullptr) {
            return std::nullopt;
        }
        return compilerPtr->get_supported_options();
    }

    if (auto cachedOptions = CompilerOptionsCache::getSupportedOptions(effectiveCompilerType);
        cachedOptions.has_value()) {
        return cachedOptions;
    }

    auto* compilerPtr = getCompiler();
    if (compilerPtr == nullptr) {
        return std::nullopt;
    }

    return compilerPtr->get_supported_options();
}
}  // namespace intel_npu
