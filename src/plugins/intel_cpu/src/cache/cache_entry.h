// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "lru_cache.h"

namespace ov {
namespace intel_cpu {

class CacheEntryBase {
public:
    enum class LookUpStatus : int8_t { Hit, Miss };

public:
    virtual ~CacheEntryBase() = default;
};

/**
 * @brief Class represents a templated record in multi cache
 * @tparam KeyType is a key type that must define hash() const method with return type convertible to size_t and define
 * comparison operator.
 * @tparam ValType is a type that must meet all the requirements to the std::unordered_map mapped type
 * @tparam ImplType is a type for the internal storage. It must provide put(KeyType, ValueType) and ValueType get(const
 * KeyType&) interface and must have constructor of type ImplType(size_t).
 *
 * @note In this implementation default constructed value objects are treated as empty objects.
 */

template <typename KeyType, typename ValType, typename ImplType = LruCache<KeyType, ValType>>
class CacheEntry : public CacheEntryBase {
public:
    using ResultType = std::pair<ValType, LookUpStatus>;

public:
    explicit CacheEntry(size_t capacity) : _impl(capacity) {}

    /**
     * @brief Searches the key in the underlying storage and returns value if it exists, or creates a value using the
     * builder functor and adds it to the underlying storage.
     * @param key is the search key
     * @param builder is a callable object that creates the ValType object from the KeyType lval reference
     * @return result of the operation which is a pair of the requested object of ValType and the status of whether the
     * cache hit or miss occurred
     */

    ResultType getOrCreate(const KeyType& key, std::function<ValType(const KeyType&)> builder) {
        if (0 == _impl.getCapacity()) {
            // fast track
            return {builder(key), CacheEntryBase::LookUpStatus::Miss};
        }
        auto retStatus = LookUpStatus::Hit;
        ValType retVal = _impl.get(key);
        auto retEmpty = ValType();
        if (retVal == retEmpty) {
            retStatus = LookUpStatus::Miss;
            retVal = builder(key);
            if (retVal != retEmpty)
                _impl.put(key, retVal);
        }
        return {retVal, retStatus};
    }

public:
    ImplType _impl;
};

}  // namespace intel_cpu
}  // namespace ov
