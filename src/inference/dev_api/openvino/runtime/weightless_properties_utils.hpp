// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <variant>
#include <filesystem>

#include "openvino/runtime/properties.hpp"

namespace ov::util{
    /**
 * @brief Check if configuration has properties to enabled weightless blob.
 *
 *
 * @param config Input configuration to examine.
 * @return truw  If ENABLE_WEIGHTLESS:true
 * @return true  If no ENABLE_WEIGHTLESS and (CACHE_DIR and CACHE_MODE:OPTIMIZE_SIZE)
 * @return false If ENABLE_WEIGHTLESS:false
 * @return false If no ENABLE_WEIGHTLESS and (CACHE_DIR and (CACHE_MODE:OPTIMIZE_SPEED or no CACHE_MODE)
 */
inline bool is_weightless_enabled(const ov::AnyMap& config) {
    if (auto it = config.find(ov::enable_weightless.name()); it != config.end()) {
        return it->second.as<bool>();
    } else if (auto cache_mode_it = config.find(ov::cache_mode.name()); cache_mode_it != config.end()) {
        return cache_mode_it->second.as<ov::CacheMode>() == ov::CacheMode::OPTIMIZE_SIZE;
    }
    return false;
}

using WeightlessHint = std::variant<std::filesystem::path, std::shared_ptr<const ov::Model>, ov::Tensor>;

// Extract weightless hint from config as variant which can be used by plugin to apply specific logic to restore weights
/**
 * @brief Get the weightless hint object
 *
 * @param config
 * @return WeightlessHint
*/
inline WeightlessHint get_weightless_hint(const ov::AnyMap& config) {
    WeightlessHint hint;
    if (auto it = config.find(ov::weights_path.name()); it != config.end()) {
        hint = std::filesystem::path(it->second.as<std::string>());
    } else if (auto it = config.find(ov::hint::model.name()); it != config.end()) {
        hint = it->second.as<std::shared_ptr<const ov::Model>>();
    } else if (auto it = config.find(ov::hint::compiled_blob.name()); it != config.end()) {
        hint = it->second.as<Tensor>();
    }
    return hint;
}
}
