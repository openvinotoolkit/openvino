// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ov {
namespace npuw {
namespace s11n {
using IndicatorType = std::array<uint8_t, 6>;
}  // namespace s11n
}  // namespace npuw
}  // namespace ov

const constexpr ov::npuw::s11n::IndicatorType NPUW_SERIALIZATION_INDICATOR =
    {char{0x13}, char{0x37}, char{0x6e}, char{0x70}, char{0x75}, char{0x77}};

const constexpr char* NPUW_SERIALIZATION_VERSION = "0.2";

// Forward declaration
namespace intel_npu {
class Config;
}  // namespace intel_npu

namespace ov {

// Forward declaration
class Any;
class Node;
class Tensor;
template <class>
class Output;
template <class>
class SharedBuffer;
class MappedMemory;
enum class CacheMode;
namespace element {
class Type;
}

// Forward declaration
namespace op {
namespace v0 {
class Parameter;
}  // namespace v0
}  // namespace op

namespace npuw {

// Forward declaration
namespace compiled {
struct Spatial;
}  // namespace compiled
namespace weights {
class LazyTensor;
}  // namespace weights

namespace s11n {

struct Context {
    explicit Context(bool _is_weightless, const std::unordered_map<const void*, std::size_t>& _const_to_offset)
        : is_weightless(_is_weightless),
          const_to_offset(_const_to_offset) {}
    bool is_weightless;
    const std::unordered_map<const void*, std::size_t>& const_to_offset;
};

struct LLMSerializeContext {
    explicit LLMSerializeContext(bool _encrypted, std::function<std::string(const std::string&)> _encrypt)
        : encrypted(_encrypted),
          encrypt(_encrypt) {}
    bool encrypted = false;
    std::function<std::string(const std::string&)> encrypt = nullptr;
};

struct LLMDeserializeContext {
    explicit LLMDeserializeContext(bool _encrypted, std::function<std::string(const std::string&)> _decrypt)
        : encrypted(_encrypted),
          decrypt(_decrypt) {}
    bool encrypted = false;
    std::function<std::string(const std::string&)> decrypt = nullptr;
};

using Weights = std::shared_ptr<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>;

// Specific type overloads
void write(std::ostream& stream, const std::streampos& var);
void write(std::ostream& stream, const std::string& var);
void write(std::ostream& stream, const bool& var);
void write(std::ostream& stream, const float& var);
void write(std::ostream& stream, const ov::npuw::compiled::Spatial& var);
void write(std::ostream& stream, const ov::Tensor& var);
void write(std::ostream& stream, const ::intel_npu::Config& var);
void write(std::ostream& stream, const ov::Output<const ov::Node>& var);
void write_any(std::ostream& stream, const ov::Any& var);
void write(std::ostream& stream, const ov::npuw::weights::LazyTensor& var);
void write(std::ostream& stream, const ov::CacheMode& var);
void write(std::ostream& stream, const ov::element::Type& var);

void read(std::istream& stream, std::streampos& var);
void read(std::istream& stream, std::string& var);
void read(std::istream& stream, bool& var);
void read(std::istream& stream, float& var);
void read(std::istream& stream, ov::npuw::compiled::Spatial& var);
void read(std::istream& stream, ov::Tensor& var);
void read(std::istream& stream, ::intel_npu::Config& var);
void read(std::istream& stream, std::shared_ptr<ov::op::v0::Parameter>& var);
void read(std::istream& stream, std::shared_ptr<ov::Node>& var);
void read_any(std::istream& stream, ov::Any& var);
void read(std::istream& stream, ov::npuw::weights::LazyTensor& var);
void read(std::istream& stream, ov::CacheMode& var);
void read(std::istream& stream, ov::element::Type& var);

// Weightless utils
void write_weightless(std::ostream& stream, const std::vector<ov::Tensor>& var, const Context& ctx);
// No allocation needed
void read_weightless(std::istream& stream, std::vector<ov::Tensor>& var, const Weights& weights);

// Forward declaration
template <typename T1, typename T2>
void write(std::ostream& stream, const std::pair<T1, T2>& var);
template <typename T>
void write(std::ostream& stream, const std::vector<T>& var);
template <typename T, size_t N>
void write(std::ostream& stream, const std::array<T, N>& var);
template <typename T1, typename T2>
void read(std::istream& stream, std::pair<T1, T2>& var);
template <typename T>
void read(std::istream& stream, std::vector<T>& var);
template <typename T, std::size_t N>
void read(std::istream& stream, std::array<T, N>& var);

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

template <typename T, size_t N>
void write(std::ostream& stream, const std::array<T, N>& var) {
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
    var.clear();
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    var.reserve(var_size);
    for (std::size_t i = 0; i < var_size; ++i) {
        T elem;
        read(stream, elem);
        var.push_back(std::move(elem));
    }
}

template <typename T, std::size_t N>
void read(std::istream& stream, std::array<T, N>& var) {
    for (std::size_t i = 0; i < N; ++i) {
        T elem;
        read(stream, elem);
        var[i] = elem;
    }
}

template <typename T>
void read(std::istream& stream, std::unordered_set<T>& var) {
    var.clear();
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    for (std::size_t i = 0; i < var_size; ++i) {
        T elem;
        read(stream, elem);
        var.insert(std::move(elem));
    }
}

template <typename K, typename V>
void read(std::istream& stream, std::map<K, V>& var) {
    var.clear();
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
