// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "log/debug.hpp"

namespace ov {
namespace intel_gna {
namespace common {

template <typename T, typename U>
U GetValueForKey(const T& key, const std::unordered_map<T, U>& mapping) {
    const auto key_iter = mapping.find(key);
    if (key_iter != mapping.end()) {
        return key_iter->second;
    }
    THROW_GNA_EXCEPTION << "Unsupported map key" << std::endl;
}

template <typename T, typename U>
T GetKeyForValue(const U& value, const std::unordered_map<T, U>& mapping) {
    for (const auto& item : mapping) {
        if (item.second == value) {
            return item.first;
        }
    }
    THROW_GNA_EXCEPTION << "Unsupported map value" << std::endl;
}

}  // namespace common
}  // namespace intel_gna
}  // namespace ov
