// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include "lru_cache.h"

namespace MKLDNNPlugin {

class CacheBase {
public:
    virtual ~CacheBase() = default;
};

template<typename Key, typename Value, typename Impl = LruCache<Key, Value>>
class ExecutorCache : public CacheBase {
public:
    using Builder = std::function<Value(const Key&)>;

public:
    ExecutorCache(size_t capacity, Builder builder) : _impl(capacity), _builder(std::move(builder)) {}
    Value getOrCreate(const Key& key) {
        auto retVal = _impl.get(key);
        if (retVal == Value()) {
            //cache miss
            retVal = _builder(key);
            _impl.put(key, retVal);
        }
        return retVal;
    }

private:
    Impl _impl;
    Builder _builder;
};

} // namespace MKLDNNPlugin