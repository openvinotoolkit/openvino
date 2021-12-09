// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>
#include "lru_cache.h"

namespace MKLDNNPlugin {

class CacheEntryBase {
public:
    enum class LookUpStatus : int8_t {
        Hit,
        Miss
    };
public:
    virtual ~CacheEntryBase() = default;
};

template<typename KeyType,
         typename ValType,
         typename ImplType = LruCache<KeyType, ValType>>
class CacheEntry : public CacheEntryBase {
public:
    using ResultType = std::pair<ValType, LookUpStatus>;
public:
    explicit CacheEntry(size_t capacity) : _impl(capacity) {}
    ResultType getOrCreate(const KeyType& key, std::function<ValType(const KeyType&)> builder) {
        auto retStatus = LookUpStatus::Hit;
        ValType retVal = _impl.get(key);
        if (retVal == ValType()) {
            retStatus = LookUpStatus::Miss;
            retVal = builder(key);
            _impl.put(key, retVal);
        }
        return {retVal, retStatus};
    }

public:
    ImplType _impl;
};
}// namespace MKLDNNPlugin
