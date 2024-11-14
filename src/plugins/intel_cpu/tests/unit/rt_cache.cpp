// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "cache/lru_cache.h"
#include "cache/multi_cache.h"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::intel_cpu;

namespace {
struct IntKey {
    size_t hash() const {
        return std::hash<int>().operator()(data);
    }
    bool operator==(const IntKey& rhs) const noexcept {
        return this->data == rhs.data;
    }

    int data;
};
} // namespace

TEST(LruCacheTests, Evict) {
    constexpr size_t capacity = 10;
    LruCache<IntKey, int> cache(capacity);
    for (size_t i = 0; i < 2 * capacity; ++i) {
        OV_ASSERT_NO_THROW(cache.put({10}, 10));
    }
    OV_ASSERT_NO_THROW(cache.evict(5));
    OV_ASSERT_NO_THROW(cache.evict(10));
    int result = cache.get({10});
    ASSERT_EQ(result, int());
    OV_ASSERT_NO_THROW(cache.evict(0));
}

TEST(LruCacheTests, Put) {
    constexpr size_t capacity = 10;
    LruCache<IntKey, int> cache(capacity);
    for (size_t i = 0; i < 2 * capacity; ++i) {
        OV_ASSERT_NO_THROW(cache.put({10}, 10));
    }

    ASSERT_EQ(cache.get({10}), 10);
}

TEST(LruCacheTests, Get) {
    constexpr int capacity = 10;
    LruCache<IntKey, int> cache(capacity);
    for (int i = 1; i < 2 * capacity; ++i) {
        OV_ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 1; i < capacity; ++i) {
        ASSERT_EQ(cache.get({i}), int());
    }

    for (int i = capacity; i < 2 * capacity; ++i) {
        ASSERT_EQ(cache.get({i}), i);
    }
}

TEST(LruCacheTests, LruPolicy) {
    constexpr int capacity = 10;
    LruCache<IntKey, int> cache(capacity);
    for (int i = 1; i < capacity; ++i) {
        OV_ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 4; i < capacity; ++i) {
        ASSERT_EQ(cache.get({i}), i);
    }

    for (int i = 21; i < 25; ++i) {
        OV_ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 1; i < 4; ++i) {
        ASSERT_EQ(cache.get({i}), int());
    }
}

TEST(LruCacheTests, Empty) {
    constexpr size_t capacity = 0;
    constexpr int attempts = 10;
    LruCache<IntKey, int> cache(capacity);
    for (int i = 1; i < attempts; ++i) {
        OV_ASSERT_NO_THROW(cache.put({i}, i));
    }

    for (int i = 1; i < attempts; ++i) {
        ASSERT_EQ(cache.get({i}), int());
    }
}
namespace {
template<typename T, typename K>
class mockBuilder {
public:
    MOCK_METHOD(T, build, (const K&));
};
}// namespace

TEST(CacheEntryTests, GetOrCreate) {
    using testing::_;
    using ValueType = std::shared_ptr<int>;

    constexpr int capacity = 10;

    mockBuilder<ValueType::element_type, IntKey> builderMock;
    EXPECT_CALL(builderMock, build(_))
            .Times(3 * capacity)
            .WillRepeatedly([](const IntKey& key){return key.data;});

    auto builder = [&](const IntKey& key) { return std::make_shared<int>(builderMock.build(key)); };

    CacheEntry<IntKey, ValueType> entry(capacity);

    //creating so we miss everytime
    for (int i = 0; i < capacity; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //always hit
    for (int i = 0; i < capacity; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Hit);
    }

    //new values displace old ones
    for (int i = capacity; i < 2 * capacity; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //can not hit the old ones
    for (int i = 0; i < capacity; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Miss);
    }
}

TEST(CacheEntryTests, Empty) {
    using testing::_;
    using ValueType = std::shared_ptr<int>;

    constexpr size_t capacity = 0;
    constexpr int attempts = 10;

    mockBuilder<ValueType::element_type, IntKey> builderMock;
    EXPECT_CALL(builderMock, build(_))
            .Times(2 * attempts)
            .WillRepeatedly([](const IntKey& key){return key.data;});

    auto builder = [&](const IntKey& key) { return std::make_shared<int>(builderMock.build(key)); };

    CacheEntry<IntKey, ValueType> entry(capacity);

    //creating so we miss everytime
    for (int i = 0; i < attempts; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //since the capacity is 0 we will always miss
    for (int i = 0; i < attempts; ++i) {
        auto result = entry.getOrCreate({i}, builder);
        ASSERT_NE(result.first, ValueType());
        ASSERT_EQ(*result.first, i);
        ASSERT_EQ(result.second, CacheEntryBase::LookUpStatus::Miss);
    }
}

namespace {
struct StringKey {
    size_t hash() const {
        return std::hash<std::string>().operator()(data);
    }
    bool operator==(const StringKey& rhs) const noexcept {
        return this->data == rhs.data;
    }

    std::string data;
};
} // namespace

TEST(MultiCacheTests, GetOrCreate) {
    using testing::_;
    using IntValueType = std::shared_ptr<int>;
    using StrValueType = std::shared_ptr<std::string>;

    constexpr int capacity = 10;

    mockBuilder<IntValueType::element_type, IntKey> intBuilderMock;
    EXPECT_CALL(intBuilderMock, build(_))
            .Times(3 * capacity)
            .WillRepeatedly([](const IntKey& key){return key.data;});

    mockBuilder<StrValueType::element_type, StringKey> strBuilderMock;
    EXPECT_CALL(strBuilderMock, build(_))
            .Times(3 * capacity)
            .WillRepeatedly([](const StringKey& key){return key.data;});

    auto intBuilder = [&](const IntKey& key) { return std::make_shared<int>(intBuilderMock.build(key)); };
    auto strBuilder = [&](const StringKey& key) { return std::make_shared<std::string>(strBuilderMock.build(key)); };

    MultiCache cache(capacity);

    //creating so we miss everytime
    for (int i = 0; i < capacity; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //always hit
    for (int i = 0; i < capacity; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Hit);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Hit);
    }

    //new values displace old ones
    for (int i = capacity; i < 2 * capacity; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //can not hit the old ones
    for (int i = 0; i < capacity; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
    }
}

TEST(MultiCacheTests, Empty) {
    using testing::_;
    using IntValueType = std::shared_ptr<int>;
    using StrValueType = std::shared_ptr<std::string>;

    constexpr size_t capacity = 0;
    constexpr int attempts = 10;

    mockBuilder<IntValueType::element_type, IntKey> intBuilderMock;
    EXPECT_CALL(intBuilderMock, build(_))
            .Times(2 * attempts)
            .WillRepeatedly([](const IntKey& key){return key.data;});

    mockBuilder<StrValueType::element_type, StringKey> strBuilderMock;
    EXPECT_CALL(strBuilderMock, build(_))
            .Times(2 * attempts)
            .WillRepeatedly([](const StringKey& key){return key.data;});

    auto intBuilder = [&](const IntKey& key) { return std::make_shared<int>(intBuilderMock.build(key)); };
    auto strBuilder = [&](const StringKey& key) { return std::make_shared<std::string>(strBuilderMock.build(key)); };

    MultiCache cache(capacity);

    //creating so we miss everytime
    for (int i = 0; i < attempts; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
    }

    //since the capacity is 0 we will always miss
    for (int i = 0; i < attempts; ++i) {
        auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
        ASSERT_NE(intResult.first, IntValueType());
        ASSERT_EQ(*intResult.first, i);
        ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
        auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
        ASSERT_NE(strResult.first, StrValueType());
        ASSERT_EQ(*strResult.first, std::to_string(i));
        ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
    }
}

namespace {
class ScopedThread {
public:
    explicit ScopedThread(std::thread t) : _t(std::move(t)) {
        if (!_t.joinable()) {
            std::logic_error("Thread is not joinable!");
        }
    }
    ~ScopedThread() {
        _t.join();
    }
    ScopedThread(ScopedThread&& rhs) noexcept = default;
private:
    std::thread _t;
};
}// namespace


TEST(MultiCacheTests, SmokeTypeIdSync) {
    using IntValueType = std::shared_ptr<int>;
    using StrValueType = std::shared_ptr<std::string>;

    constexpr int capacity = 10;
    constexpr size_t numThreads = 30;

    auto intBuilder = [&](const IntKey& key) { return std::make_shared<int>(key.data); };
    auto strBuilder = [&](const StringKey& key) { return std::make_shared<std::string>(key.data); };

    std::vector<MultiCache> vecCache(numThreads, MultiCache(capacity));

    auto testRoutine = [&](MultiCache& cache) {
        //creating so we miss everytime
        for (int i = 0; i < capacity; ++i) {
            auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
            ASSERT_NE(intResult.first, IntValueType());
            ASSERT_EQ(*intResult.first, i);
            ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Miss);
            auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
            ASSERT_NE(strResult.first, StrValueType());
            ASSERT_EQ(*strResult.first, std::to_string(i));
            ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Miss);
        }

        //always hit
        for (int i = 0; i < capacity; ++i) {
            auto intResult = cache.getOrCreate(IntKey{i}, intBuilder);
            ASSERT_NE(intResult.first, IntValueType());
            ASSERT_EQ(*intResult.first, i);
            ASSERT_EQ(intResult.second, CacheEntryBase::LookUpStatus::Hit);
            auto strResult = cache.getOrCreate(StringKey{std::to_string(i)}, strBuilder);
            ASSERT_NE(strResult.first, StrValueType());
            ASSERT_EQ(*strResult.first, std::to_string(i));
            ASSERT_EQ(strResult.second, CacheEntryBase::LookUpStatus::Hit);
        }
    };

    std::vector<ScopedThread> vecThreads;
    vecThreads.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        vecThreads.emplace_back(std::thread(testRoutine, std::ref(vecCache[i])));
    }
}
