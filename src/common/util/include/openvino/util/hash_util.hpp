// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace ov::util {

/**
 * @brief Combines two hash values.
 * @param val   New hash value to combine.
 * @param seed  Existing hash seed.
 * @return Combined hash value.
 */
inline size_t hash_combine(size_t val, const size_t seed) {
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Combines a vector of hash values into a single hash.
 * @param list  Vector of hash values.
 * @return Combined hash value.
 */
inline size_t hash_combine(const std::vector<size_t>& list) {
    size_t seed = 0;
    for (size_t v : list) {
        seed ^= hash_combine(v, seed);
    }
    return seed;
}

/**
 * @brief Combines an initializer list of hash values into a single hash.
 * @param list  Initializer list of hash values.
 * @return Combined hash value.
 */
inline size_t hash_combine(std::initializer_list<size_t>&& list) {
    size_t seed = 0;
    for (size_t v : list) {
        seed ^= hash_combine(v, seed);
    }
    return seed;
}

/**
 * @brief Combines two 64-bit hash values using Murmur-inspired mixing.
 * @param h  Existing hash seed.
 * @param k  New value to mix in.
 * @return Combined 64-bit hash.
 */
constexpr uint64_t u64_hash_combine(uint64_t h, uint64_t k) {
    // Hash combine formula from boost for uint64_t.
    constexpr uint64_t m = 0xc6a4a7935bd1e995;
    constexpr int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    return h + 0xe6546b64;
}

/**
 * @brief Combines a seed with a list of 64-bit hash values.
 * @param seed    Initial seed.
 * @param values  Values to combine.
 * @return Combined 64-bit hash.
 */
constexpr uint64_t u64_hash_combine(uint64_t seed, std::initializer_list<uint64_t>&& values) {
    uint64_t h = seed;
    for (uint64_t k : values) {
        h = u64_hash_combine(h, k);
    }
    return h;
}

}  // namespace ov::util
