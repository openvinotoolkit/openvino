// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <functional>
#include <unordered_map>

#include "cache_entry.h"

namespace ov {
namespace intel_cpu {

/**
 * @brief Class that represent a preemptive cache for different key/value pair types.
 *
 * @attention This implementation IS NOT THREAD SAFE!
 */

class MultiCache {
public:
    template <typename KeyType, typename ValueType>
    using EntryTypeT = CacheEntry<KeyType, ValueType>;
    using EntryBasePtr = std::shared_ptr<CacheEntryBase>;
    template <typename KeyType, typename ValueType>
    using EntryPtr = std::shared_ptr<EntryTypeT<KeyType, ValueType>>;

public:
    /**
     * @param capacity here means maximum records limit FOR EACH entry specified by a pair of Key/Value types.
     * @note zero capacity means empty cache so no records are stored and no entries are created
     */
    explicit MultiCache(size_t capacity) : _capacity(capacity) {}

    /**
     * @brief Searches a value of ValueType in the cache using the provided key or creates a new ValueType instance (if
     * nothing was found) using the key and the builder functor and adds the new record to the cache
     * @param key is the search key
     * @param builder is a callable object that creates the ValType object from the KeyType lval reference.
     *       Also the builder type is used for the ValueType deduction
     * @return result of the operation which is a pair of the requested object of ValType and the status of whether the
     * cache hit or miss occurred
     */
    template <typename KeyType,
              typename BuilderType,
              typename ValueType = std::invoke_result_t<BuilderType&, const KeyType&>>
    typename CacheEntry<KeyType, ValueType>::ResultType getOrCreate(const KeyType& key, BuilderType builder) {
        auto entry = getEntry<KeyType, ValueType>();
        return entry->getOrCreate(key, std::move(builder));
    }

private:
    template <typename T>
    size_t getTypeId();
    template <typename KeyType, typename ValueType>
    EntryPtr<KeyType, ValueType> getEntry();

private:
    static std::atomic_size_t _typeIdCounter;
    size_t _capacity;
    std::unordered_map<size_t, EntryBasePtr> _storage;
};

template <typename T>
size_t MultiCache::getTypeId() {
    static size_t id = _typeIdCounter.fetch_add(1);
    return id;
}

template <typename KeyType, typename ValueType>
MultiCache::EntryPtr<KeyType, ValueType> MultiCache::getEntry() {
    using EntryType = EntryTypeT<KeyType, ValueType>;
    size_t id = getTypeId<EntryType>();
    auto itr = _storage.find(id);
    if (itr == _storage.end()) {
        auto result = _storage.insert({id, std::make_shared<EntryType>(_capacity)});
        itr = result.first;
    }
    return std::static_pointer_cast<EntryType>(itr->second);
}

using MultiCacheWeakPtr = std::weak_ptr<MultiCache>;
using MultiCacheWeakCPtr = std::weak_ptr<const MultiCache>;
using MultiCachePtr = std::shared_ptr<MultiCache>;
using MultiCacheCPtr = std::shared_ptr<const MultiCache>;

}  // namespace intel_cpu
}  // namespace ov
