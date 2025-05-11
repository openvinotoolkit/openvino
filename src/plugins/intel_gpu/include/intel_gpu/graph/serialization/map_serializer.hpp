// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <type_traits>
#include "buffer.hpp"

namespace cldnn {

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::map<Key, Value>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::map<Key, Value>& map) {
        buffer << map.size();
        for (const auto& pair : map) {
            buffer(pair.first, pair.second);
        }
    }
};

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::map<Key, Value>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::map<Key, Value>& map) {
        typename std::map<Key, Value>::size_type map_size = 0UL;
        buffer >> map_size;
        map.clear();
        Key key;
        for (size_t i = 0; i < map_size; i++) {
            buffer >> key;
            buffer >> map[std::move(key)];
        }
    }
};

}  // namespace cldnn
