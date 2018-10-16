// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef HASH_HPP
#define HASH_HPP

#include <cstddef> //size_t

namespace util
{
inline std::size_t hash_combine(std::size_t seed, std::size_t val)
{
    // Hash combine formula from boost
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
} // namespace util

#endif // HASH_HPP
