// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"

namespace cldnn {
template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                    !std::is_same<bool, T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size(); //static_cast<uint64_t>()
        buffer << make_data(vector.data(), static_cast<uint64_t>(vector.size() * sizeof(T)));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                     !std::is_same<bool, T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        typename std::vector<T>::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        buffer >> make_data(vector.data(), static_cast<uint64_t>(vector_size * sizeof(T)));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                    !std::is_arithmetic<T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                                    !std::is_arithmetic<T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        typename std::vector<T>::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};

}  // namespace cldnn
