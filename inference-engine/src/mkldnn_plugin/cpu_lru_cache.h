// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <unordered_map>

/**
 * @brief This is yet another implementation of a cache with limited capacity and LRU eviction policy.
 * @tparam Key is a key type that must define hash() const method with return type convertible to size_t and define comparison operator.
 * @tparam Value is a type that must meet all the requirements to the std::unordered_map mapped type
 *
 * @attention This cache implementation IS NOT THREAD SAFE!
 */

namespace MKLDNNPlugin {

template<typename Key, typename Value>
class LruCache {
public:
    using value_type = std::pair<Key, Value>;

public:
    explicit LruCache(size_t capacity) : _capacity(capacity) {}

    void put(Key key, Value val) {
        auto mapItr = _cacheMapper.find(key);
        if (mapItr != _cacheMapper.end()) {
            touch(mapItr->second);
            mapItr->second->second = val;
        } else {
            if (_cacheMapper.size() == _capacity) {
                evict(1);
            }
            auto itr = _lruList.insert(_lruList.begin(), {key, std::move(val)});
            _cacheMapper.insert({std::move(key), itr});
        }
    }

    Value get(const Key &key) {
        auto itr = _cacheMapper.find(key);
        if (itr == _cacheMapper.end()) {
            return Value();
        }

        touch(itr->second);
        return _lruList.front().second;
    }

    void evict(size_t n) {
        for (size_t i = 0; i < n && !_lruList.empty(); ++i) {
            _cacheMapper.erase(_lruList.back().first);
            _lruList.pop_back();
        }
    }

private:
    struct key_hasher {
        std::size_t operator()(const Key &k) const {
            return k.hash();
        }
    };

    using lru_list_type = std::list<value_type>;
    using cache_map_value_type = typename lru_list_type::iterator;

    void touch(typename lru_list_type::iterator itr) {
        _lruList.splice(_lruList.begin(), _lruList, itr);
    }

    lru_list_type _lruList;
    std::unordered_map<Key, cache_map_value_type, key_hasher> _cacheMapper;
    size_t _capacity;
};

} // namespace MKLDNNPlugin