// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "cpu_lru_cache.h"

namespace {
struct Key {
    size_t hash() const {
        return std::hash<int>().operator()(data);
    }
    bool operator==(const Key& rhs) const noexcept {
        return this->data == rhs.data;
    }

    int data;
};
} // namespace

TEST(LruCacheTests, Evict) {
    constexpr size_t capacity = 10;
    MKLDNNPlugin::LruCache<Key, int> cache(capacity);
    for (size_t i = 0; i < 2 * capacity; ++i) {
        ASSERT_NO_THROW(cache.put({10}, 10));
    }
    ASSERT_NO_THROW(cache.evict(5));
    ASSERT_NO_THROW(cache.evict(10));
    int result = cache.get({10});
    ASSERT_EQ(result, 0);
    ASSERT_NO_THROW(cache.evict(0));
}

TEST(LruCacheTests, Put) {
    constexpr size_t capacity = 10;
    MKLDNNPlugin::LruCache<Key, int> cache(capacity);
    for (size_t i = 0; i < 2 * capacity; ++i) {
        ASSERT_NO_THROW(cache.put({10}, 10));
    }

    ASSERT_EQ(cache.get({10}), 10);
}

TEST(LruCacheTests, Get) {
    constexpr size_t capacity = 10;
    MKLDNNPlugin::LruCache<Key, int> cache(capacity);
    for (int i = 1; i < 2 * capacity; ++i) {
        ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 1; i < capacity; ++i) {
        ASSERT_EQ(cache.get({i}), 0);
    }

    for (int i = capacity; i < 2 * capacity; ++i) {
        ASSERT_EQ(cache.get({i}), i);
    }
}

TEST(LruCacheTests, LruPolicy) {
    constexpr size_t capacity = 10;
    MKLDNNPlugin::LruCache<Key, int> cache(capacity);
    for (int i = 1; i < capacity; ++i) {
        ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 4; i < capacity; ++i) {
        ASSERT_EQ(cache.get({i}), i);
    }

    for (int i = 21; i < 25; ++i) {
        ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 1; i < 4; ++i) {
        ASSERT_EQ(cache.get({i}), 0);
    }
}
