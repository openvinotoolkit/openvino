// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

namespace ov::intel_cpu::hash {

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T, std::enable_if_t<!std::is_enum_v<T>, int> = 0>
size_t combine(size_t seed, const T& v) {
    return seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
size_t combine(size_t seed, const T& v) {
    using underlying_t = std::underlying_type_t<T>;
    return combine(seed, static_cast<underlying_t>(v));
}

template <typename T>
size_t combine(size_t seed, const std::vector<T>& v) {
    for (const auto& elem : v) {
        seed = combine(seed, elem);
    }
    return seed;
}

struct Builder {
    Builder(size_t seed) : m_seed(seed) {}

    // todo add specializations / sfinae
    template <typename T>
    Builder& combine(T v) {
        m_seed = ov::intel_cpu::hash::combine(m_seed, v);
        return *this;
    }

    size_t generate() {
        return m_seed;
    }

private:
    size_t m_seed;
};

}  // namespace ov::intel_cpu::hash
