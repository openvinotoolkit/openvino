// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <unordered_set>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "openvino/core/axis_set.hpp"

namespace cldnn {
template <typename BufferType, typename T>
class Serializer<BufferType, std::set<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::set<T>& set) {
        buffer << set.size();
        for (const auto& el : set) {
            buffer << el;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::set<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::set<T>& set) {
        typename std::set<T>::size_type set_size = 0UL;
        buffer >> set_size;

        for (long unsigned int i = 0; i < set_size; i++) {
            T el;
            buffer >> el;
            set.insert(el);
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::unordered_set<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::unordered_set<T>& set) {
        buffer << set.size();
        for (const auto& el : set) {
            buffer << el;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::unordered_set<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::unordered_set<T>& set) {
        typename std::unordered_set<T>::size_type set_size = 0UL;
        buffer >> set_size;

        for (long unsigned int i = 0; i < set_size; i++) {
            T el;
            buffer >> el;
            set.insert(el);
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::AxisSet, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::AxisSet& set) {
        buffer << set.size();
        for (const auto& el : set) {
            buffer << el;
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::AxisSet, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::AxisSet& set) {
        typename ov::AxisSet::size_type set_size = 0UL;
        buffer >> set_size;

        for (long unsigned int i = 0; i < set_size; i++) {
            size_t el;
            buffer >> el;
            set.insert(el);
        }
    }
};
}  // namespace cldnn
