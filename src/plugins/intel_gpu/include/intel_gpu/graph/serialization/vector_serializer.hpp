// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

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

template <typename BufferType>
class Serializer<BufferType, std::vector<bool>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<bool>& vector) {
        buffer << vector.size();
        for (const bool el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, std::vector<bool>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<bool>& vector) {
        typename std::vector<bool>::size_type vector_size = 0UL;
        buffer >> vector_size;
        bool el;
        vector.clear();
        for (size_t i = 0; i < vector_size; ++i) {
            buffer >> el;
            vector.push_back(el);
        }
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

template <typename BufferType>
class Serializer<BufferType, ov::CoordinateDiff, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::CoordinateDiff& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::CoordinateDiff, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::CoordinateDiff& vector) {
        typename ov::CoordinateDiff::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::Strides, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::Strides& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::Strides, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::Strides& vector) {
        typename ov::Strides::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::Shape, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::Shape& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::Shape, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::Shape& vector) {
        typename ov::Shape::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};

}  // namespace cldnn
