// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/common/option_support_cache.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class CompilerOptionSupportHelper final {
public:
    explicit CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend,
                                         const std::shared_ptr<const CompilerAdapterFactory>& adapterFactory,
                                         const std::shared_ptr<OptionSupportCache>& optionSupportCache);

    bool isOptionSupported(ov::intel_npu::CompilerType compilerType,
                           const std::string& optionName,
                           const std::optional<std::string>& optionValue = std::nullopt);

private:
    const ov::SoPtr<IEngineBackend> _backend;
    const std::shared_ptr<const CompilerAdapterFactory> _adapterFactory;
    std::shared_ptr<OptionSupportCache> _optionSupportCache;

    std::mutex _supportedOptionsLoadedMutex;
    std::unordered_map<OptionSupportCache::CacheKey, std::once_flag> _supportedOptionsLoaded;
};

}  // namespace intel_npu
