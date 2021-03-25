// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <map>
#include <mutex>
#include <memory>
#include <utility>
#include <functional>

namespace cldnn {
namespace gpu {

template <typename Key,
          typename Type,
          class Traits = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, Type>>>
class push_pop_map {
    std::mutex _mutex;
    std::map<Key, Type, Traits, Allocator> _map;

public:
    void push(const Key& key, Type value) {
        std::lock_guard<std::mutex> lock{_mutex};
        _map.insert({key, std::move(value)});
    }

    Type pop(const Key& key) {
        std::lock_guard<std::mutex> lock{_mutex};
        auto it = _map.find(key);
        if (it == _map.end())
            throw std::out_of_range("Invalud push_pop_map<K, T> key");
        auto x = std::move(it->second);
        _map.erase(it);
        return std::move(x);
    }

    bool empty() {
        std::lock_guard<std::mutex> lock{_mutex};
        return _map.empty();
    }
};

}  // namespace gpu
}  // namespace cldnn
