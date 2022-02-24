// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>

#include "error.hpp"

namespace vpu {

template <template <typename, typename...> class Map,
          typename Key,
          typename Value,
          typename... AddParams>
inline std::vector<Key> getKeys(const Map<Key, Value, AddParams...>& map) {
    auto keys = std::vector<Key>{};
    keys.reserve(map.size());
    std::transform(map.cbegin(), map.cend(), std::back_inserter(keys), [](const std::pair<Key, Value>& entry) { return entry.first; });
    return keys;
}

template <template <typename, typename...> class Map,
          typename Key,
          typename Value,
          typename... AddParams>
inline std::vector<Value> getValues(const Map<Key, Value, AddParams...>& map) {
    auto values = std::vector<Value>{};
    values.reserve(map.size());
    std::transform(map.cbegin(), map.cend(), std::back_inserter(values), [](const std::pair<Key, Value>& entry) { return entry.second; });
    return values;
}

template <template <typename, typename...> class Map,
          typename Key,
          typename Value,
          typename... AddParams>
inline Map<Value, Key> inverse(const Map<Key, Value, AddParams...>& map) {
    auto inverted = Map<Value, Key>{};
    for (const auto& entry : map) {
        const auto& insertion = inverted.emplace(entry.second, entry.first);
        VPU_THROW_UNLESS(insertion.second, "Could not invert map {} due to duplicated value \"{}\"", map, entry.second);
    }
    return inverted;
}

}  // namespace vpu
