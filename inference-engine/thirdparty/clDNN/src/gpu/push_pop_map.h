/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <map>
#include <mutex>

namespace cldnn { namespace gpu {

template<typename Key, typename Type, class Traits = std::less<Key>,
    class Allocator = std::allocator<std::pair <const Key, Type> >>
    class push_pop_map {
    std::mutex _mutex;
    std::map<Key, Type, Traits, Allocator> _map;
    public:
        void push(const Key& key, Type value) {
            std::lock_guard<std::mutex> lock{ _mutex };
            _map.insert({ key, std::move(value) });
        }

        Type pop(const Key& key) {
            std::lock_guard<std::mutex> lock{ _mutex };
            auto it = _map.find(key);
            if (it == _map.end()) throw std::out_of_range("Invalud push_pop_map<K, T> key");
            auto x = std::move(it->second);
            _map.erase(it);
            return std::move(x);
        }

        bool empty() {
            std::lock_guard<std::mutex> lock{ _mutex };
            return _map.empty();
        }
};

} }
