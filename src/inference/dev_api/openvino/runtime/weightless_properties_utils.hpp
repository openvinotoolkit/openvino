// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <optional>
#include <variant>

#include "openvino/runtime/properties.hpp"

namespace ov::util {
/**
 * @brief Checks if configuration has properties which enables weightless blob.
 *
 * @param config Input configuration to examine.
 * @return std::nullopt If input configuration doesn't define to enable/disable weightless blob.
 * @return true  If ENABLE_WEIGHTLESS:true
 * @return true  If no ENABLE_WEIGHTLESS and CACHE_MODE:OPTIMIZE_SIZE
 * @return false If ENABLE_WEIGHTLESS:false
 * @return false If no ENABLE_WEIGHTLESS and CACHE_MODE:OPTIMIZE_SPEED
 */
inline std::optional<bool> is_weightless_enabled(const ov::AnyMap& config) {
    if (auto it = config.find(ov::enable_weightless.name()); it != config.end()) {
        return std::make_optional(it->second.as<bool>());
    } else if (auto cache_mode_it = config.find(ov::cache_mode.name()); cache_mode_it != config.end()) {
        return std::make_optional(cache_mode_it->second.as<ov::CacheMode>() == ov::CacheMode::OPTIMIZE_SIZE);
    }
    return std::nullopt;
}

using WeightlessHint = std::variant<std::filesystem::path, std::shared_ptr<const ov::Model>, ov::Tensor>;

/**
 * @brief Get the weightless hint from configuration.
 *
 * @param config input config to find and get weightless hint
 * @return WeightlessHint variant if exist or empty path if not found.
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
}  // namespace ov::util
