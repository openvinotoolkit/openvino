// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <unordered_map>
#include <atomic>
#include "cache_entry.h"

namespace MKLDNNPlugin {

class MultiCache {
public:
    template<typename KeyType, typename ValueType>
    using EntryTypeT = CacheEntry<KeyType, ValueType>;
    using EntryBasePtr = std::shared_ptr<CacheEntryBase>;
    template<typename KeyType, typename ValueType>
    using EntryPtr = std::shared_ptr<EntryTypeT<KeyType, ValueType>>;

public:
    explicit MultiCache(size_t capacity) : _capacity(capacity) {}
    template<typename KeyType, typename BuilderType, typename ValueType = typename std::result_of<BuilderType&(const KeyType&)>::type>
    typename CacheEntry<KeyType, ValueType>::ResultType
    getOrCreate(const KeyType& key, BuilderType builder) {
        auto entry = getEntry<KeyType, ValueType>();
        return entry->getOrCreate(key, std::move(builder));
    }

private:
    template<typename T>
    size_t getTypeId();
    template<typename KeyType, typename ValueType>
    EntryPtr<KeyType, ValueType> getEntry();

private:
    static std::atomic_size_t _typeIdCounter;
    size_t _capacity;
    std::unordered_map<size_t, EntryBasePtr> _storage;
};

template<typename T>
size_t MultiCache::getTypeId() {
    static size_t id = _typeIdCounter.fetch_add(1);
    return id;
}

template<typename KeyType, typename ValueType>
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

using MultyCachePtr = std::shared_ptr<MultiCache>;
using MultyCacheCPtr = std::shared_ptr<const MultiCache>;

} // namespace MKLDNNPlugin