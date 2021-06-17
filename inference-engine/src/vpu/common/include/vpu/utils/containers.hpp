// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>

#include "error.hpp"

namespace vpu {

template<class Key, class Value, template<class...> class Map>
inline std::vector<Key> getKeys(const Map<Key, Value>& map) {
    auto keys = std::vector<Key>{};
    keys.reserve(map.size());
    std::transform(map.cbegin(), map.cend(), std::back_inserter(keys), [](const std::pair<Key, Value>& entry) { return entry.first; });
    return keys;
}

template<class Key, class Value, template<class...> class Map>
inline std::vector<Value> getValues(const Map<Key, Value>& map) {
    auto values = std::vector<Value>{};
    values.reserve(map.size());
    std::transform(map.cbegin(), map.cend(), std::back_inserter(values), [](const std::pair<Key, Value>& entry) { return entry.second; });
    return values;
}

template<class Key, class Value, template<class...> class Map>
inline Map<Value, Key> inverse(const Map<Key, Value>& map) {
    auto inverted = Map<Value, Key>{};
    for (const auto& entry : map) {
        const auto& insertion = inverted.emplace(entry.second, entry.first);
        VPU_THROW_UNLESS(insertion.second, "Could not invert map {} due to duplicated value \"{}\"", map, entry.second);
    }
    return inverted;
}

}  // namespace vpu
