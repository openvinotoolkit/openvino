// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <string_view>

namespace intel_npu {

/**
 * @brief Helpers for string to string maps (e.g. the compiler properties). Use the option's own parser (e.g.
 * "BATCH_MODE::parse") when a typed value is needed.
 */
namespace string_map {

/**
 * @brief Checks if a value is set for the given key.
 */
inline bool has(const std::map<std::string, std::string>& map, std::string_view key) {
    return map.count(std::string(key)) != 0;
}

/**
 * @brief Retrieves the value set for the given key.
 * @return The value, or an empty string if no value was set.
 */
inline std::string get(const std::map<std::string, std::string>& map, std::string_view key) {
    const auto it = map.find(std::string(key));
    return it == map.end() ? std::string() : it->second;
}

/**
 * @brief Stores the value for the given key, overwriting a previously set one.
 */
inline void set(std::map<std::string, std::string>& map, std::string_view key, std::string value) {
    map[std::string(key)] = std::move(value);
}

}  // namespace string_map

}  // namespace intel_npu
