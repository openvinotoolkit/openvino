// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <sstream>

#include <vpu/utils/checked_cast.hpp>

namespace vpu {

std::unordered_map<int32_t, std::string> generateEnumMap(const std::string& strMap);

#define VPU_DECLARE_ENUM(EnumName, ...)                                         \
    enum class EnumName : int32_t {                                             \
        __VA_ARGS__                                                             \
    };                                                                          \
    inline std::ostream& operator<<(std::ostream& os, EnumName val) {           \
        static const auto mapName = vpu::generateEnumMap(#__VA_ARGS__);         \
        auto it = mapName.find(static_cast<int32_t>(val));                      \
        if (it != mapName.end())                                                \
            os << it->second;                                                   \
        else                                                                    \
            os << static_cast<int32_t>(val);                                    \
        return os;                                                              \
    }                                                                           \
    template <typename I>                                                       \
    inline I checked_cast(EnumName val) {                                       \
        return vpu::checked_cast<I>(static_cast<int32_t>(val));                 \
    }

struct EnumClassHash final {
    template <typename E>
    size_t operator()(E t) const {
        return std::hash<int32_t>()(static_cast<int32_t>(t));
    }
};

template <typename E>
using EnumSet = std::unordered_set<E, EnumClassHash>;

template <typename E, typename V>
using EnumMap = std::unordered_map<E, V, EnumClassHash>;

}  // namespace vpu
