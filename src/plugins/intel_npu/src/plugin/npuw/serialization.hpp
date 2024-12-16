// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "logging.hpp"

namespace ov {
namespace npuw {

// Forward declaration
namespace compiled {
struct Spatial;
}  // namespace compiled

namespace s11n {

// Specific type overloads
void write(std::ostream& stream, const std::string& var);
void write(std::ostream& stream, const bool& var);
void write(std::ostream& stream, const ov::npuw::compiled::Spatial& var);
void write(std::ostream& stream, const ov::Tensor& var);

void read(std::istream& stream, std::string& var);
void read(std::istream& stream, bool& var);
void read(std::istream& stream, ov::npuw::compiled::Spatial& var);
void read(std::istream& stream, ov::Tensor& var);

// Forward declaration
template <typename T1, typename T2>
void write(std::ostream& stream, const std::pair<T1, T2>& var);
template <typename T>
void write(std::ostream& stream, const std::vector<T>& var);
template <typename T1, typename T2>
void read(std::istream& stream, std::pair<T1, T2>& var);
template <typename T>
void read(std::istream& stream, std::vector<T>& var);

// Serialization
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void write(std::ostream& stream, const T& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

template <typename T1, typename T2>
void write(std::ostream& stream, const std::pair<T1, T2>& var) {
    write(stream, var.first);
    write(stream, var.second);
}

template <typename T>
void write(std::ostream& stream, const std::vector<T>& var) {
    write(stream, var.size());
    for (const auto& el : var) {
        write(stream, el);
    }
}

template <typename T>
void write(std::ostream& stream, const std::unordered_set<T>& var) {
    write(stream, var.size());
    for (const auto& el : var) {
        write(stream, el);
    }
}

template <typename K, typename V>
void write(std::ostream& stream, const std::map<K, V>& var) {
    write(stream, var.size());
    for (const auto& el : var) {
        write(stream, el);
    }
}

template <typename T>
void write(std::ostream& stream, const std::optional<T>& var) {
    if (var) {
        write(stream, true);
        write(stream, var.value());
    } else {
        write(stream, false);
    }
}

// Deserialization
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void read(std::istream& stream, T& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

template <typename T1, typename T2>
void read(std::istream& stream, std::pair<T1, T2>& var) {
    read(stream, var.first);
    read(stream, var.second);
}

template <typename T>
void read(std::istream& stream, std::vector<T>& var) {
    NPUW_ASSERT(var.empty() && "Can't deserialize into non-empty vector!");
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    var.reserve(var_size);
    for (std::size_t i = 0; i < var_size; ++i) {
        T elem;
        read(stream, elem);
        var.push_back(elem);
    }
}

template <typename T>
void read(std::istream& stream, std::unordered_set<T>& var) {
    NPUW_ASSERT(var.empty() && "Can't deserialize into non-empty unordered_set!");
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    for (std::size_t i = 0; i < var_size; ++i) {
        T elem;
        read(stream, elem);
        var.insert(elem);
    }
}

template <typename K, typename V>
void read(std::istream& stream, std::map<K, V>& var) {
    NPUW_ASSERT(var.empty() && "Can't deserialize into non-empty map!");
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    for (std::size_t i = 0; i < var_size; ++i) {
        std::pair<K, V> elem;
        read(stream, elem);
        var[elem.first] = elem.second;
    }
}

template <typename T>
void read(std::istream& stream, std::optional<T>& var) {
    bool has_value = false;
    read(stream, has_value);
    if (has_value) {
        T val;
        read(stream, val);
        var = val;
    }
}

}  // namespace s11n
}  // namespace npuw
}  // namespace ov
