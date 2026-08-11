// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <memory>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/compiler_supported_options_cache.hpp"

namespace intel_npu {

CompilerOptionSupportHelper::CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend)
    : _backend(backend),
      _supportedOptionsCache(std::make_shared<CompilerSupportedOptionsCache>()) {}

const std::shared_ptr<CompilerSupportedOptionsCache>& CompilerOptionSupportHelper::getCompilerSupportedOptionsCache()
    const {
    return _supportedOptionsCache;
}

bool CompilerOptionSupportHelper::isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                                    const std::string& optionName,
                                                    const std::optional<std::string>& optionValue) {
    std::unique_ptr<ICompilerAdapter> compiler;
    const auto getCompiler = [&]() -> ICompilerAdapter* {
        if (compiler == nullptr) {
            try {
                compiler = CompilerAdapterFactory().getCompiler(_backend, compilerType, "", _supportedOptionsCache);
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

    if (_supportedOptionsCache->isOptionSupported(compilerType, optionName, optionValue)) {
        return true;
    }

    auto* compilerPtr = getCompiler();
    if (compilerPtr == nullptr) {
        return false;
    }

    std::atomic<bool>& optionsLoaded = (compilerType == ov::intel_npu::CompilerType::DRIVER)
                                           ? _driverSupportedOptionsLoaded
                                           : _pluginSupportedOptionsLoaded;
    if (!optionsLoaded.exchange(true)) {
        compilerPtr->get_supported_options();
        if (_supportedOptionsCache->isOptionSupported(compilerType, optionName, optionValue)) {
            return true;
        }
    }

    return compilerPtr->is_option_supported(optionName, optionValue);
}

}  // namespace intel_npu
