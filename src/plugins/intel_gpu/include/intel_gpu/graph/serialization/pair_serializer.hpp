// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <type_traits>
#include "buffer.hpp"

namespace cldnn {

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::pair<Key, Value>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::pair<Key, Value>& pair) {
        buffer(pair.first, pair.second);
    }
};

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::pair<Key, Value>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::pair<Key, Value>& pair) {
        Key key;
        Value value;
        buffer >> key >> value;
        pair = std::make_pair(key, value);
    }
};

}  // namespace cldnn
