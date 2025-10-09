// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
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

const constexpr ov::npuw::s11n::IndicatorType NPUW_COMPILED_MODEL_INDICATOR =
    {char{0x43}, char{0x4f}, char{0x4d}, char{0x50}, char{0x4d}, char{0x4f}};

const constexpr ov::npuw::s11n::IndicatorType NPUW_LLM_COMPILED_MODEL_INDICATOR =
    {char{0x4c}, char{0x4c}, char{0x4d}, char{0x43}, char{0x4d}, char{0x4f}};

const constexpr char* NPUW_SERIALIZATION_VERSION = "0.11";

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
class Model;
enum class CacheMode;
namespace element {
class Type;
}
namespace hint {
enum class PerformanceMode;
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
struct Attention;
}  // namespace compiled
namespace weights {
class LazyTensor;
}  // namespace weights

namespace s11n {

class PairHash {
public:
    template <typename T, typename U>
    std::size_t operator()(const std::pair<T, U>& p) const {
        return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
    }
};

using BF16Cache = std::unordered_set<std::pair<std::size_t, std::size_t>, ov::npuw::s11n::PairHash>;
using Weights = ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>;
using WeightsPtr = std::shared_ptr<Weights>;

struct CompiledContext {
    CompiledContext(bool _encrypted,
                    const std::function<std::string(const std::string&)>& _encrypt,
                    const std::function<std::string(const std::string&)>& _decrypt,
                    const BF16Cache& _bf16_consts = {})
        : encrypted(_encrypted),
          encrypt(_encrypt),
          decrypt(_decrypt),
          bf16_consts(_bf16_consts) {}
    bool encrypted = false;
    std::function<std::string(const std::string&)> encrypt = nullptr;
    std::function<std::string(const std::string&)> decrypt = nullptr;
    // FIXME: needed to pass original bf16 consts meta to CompiledModel
    BF16Cache bf16_consts;
};

struct WeightsContext {
    struct CtxHash {
        inline size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
            return (std::hash<std::size_t>()(p.first) + 0x9e3779b9) ^ (std::hash<std::size_t>()(p.second) + 0x9e3779b9);
        }
    };
    using ConstsCache = std::unordered_map<std::pair<std::size_t, std::size_t>, std::shared_ptr<ov::Node>, CtxHash>;

    WeightsContext() = default;

    // NOTE: This construtor should only be used when exporting blobs
    WeightsContext(bool _is_weightless, const std::unordered_map<const void*, std::size_t>& _const_to_offset);

    // NOTE: This construtor can and should only be used when importing weightless blobs
    WeightsContext(const ov::npuw::s11n::WeightsPtr& _weights,
                   const std::string& _weights_path,
                   const ConstsCache& _consts_cache,
                   const BF16Cache& _bf16_consts);

    WeightsContext& operator=(const WeightsContext& other) = default;

    void reset() {
        weights = nullptr;
        consts_cache.clear();
    }

    bool is_weightless = true;
    std::unordered_map<const void*, std::size_t> const_to_offset;
    ov::npuw::s11n::WeightsPtr weights = nullptr;
    std::string weights_path;
    ConstsCache consts_cache;
    BF16Cache bf16_consts;
};

BF16Cache get_bf16_consts(const std::shared_ptr<ov::Model>& model);

// Specific type overloads
void write(std::ostream& stream, const std::streampos& var);
void write(std::ostream& stream, const std::string& var);
void write(std::ostream& stream, const bool& var);
void write(std::ostream& stream, const float& var);
void write(std::ostream& stream, const ov::npuw::compiled::Spatial& var);
void write(std::ostream& stream, const ov::npuw::compiled::Attention& var);
void write(std::ostream& stream, const ov::Tensor& var);
void write(std::ostream& stream, const ::intel_npu::Config& var);
void write(std::ostream& stream, const ov::Output<const ov::Node>& var);
void write_any(std::ostream& stream, const ov::Any& var);
void write(std::ostream& stream, const ov::npuw::weights::LazyTensor& var);
void write(std::ostream& stream, const ov::CacheMode& var);
void write(std::ostream& stream, const ov::element::Type& var);
void write(std::ostream& stream, const std::map<std::string, Any>& var);
void write(std::ostream& stream, const ov::hint::PerformanceMode& var);

void read(std::istream& stream, std::streampos& var);
void read(std::istream& stream, std::string& var);
void read(std::istream& stream, bool& var);
void read(std::istream& stream, float& var);
void read(std::istream& stream, ov::npuw::compiled::Spatial& var);
void read(std::istream& stream, ov::npuw::compiled::Attention& var);
void read(std::istream& stream, ov::Tensor& var);
void read(std::istream& stream, ::intel_npu::Config& var);
void read(std::istream& stream, std::shared_ptr<ov::op::v0::Parameter>& var);
void read(std::istream& stream, std::shared_ptr<ov::Node>& var);
void read_any(std::istream& stream, ov::Any& var);
void read(std::istream& stream, ov::npuw::weights::LazyTensor& var);
void read(std::istream& stream, ov::CacheMode& var);
void read(std::istream& stream, ov::element::Type& var);
void read(std::istream& stream, std::map<std::string, Any>& var);
void read(std::istream& stream, ov::hint::PerformanceMode& var);

// Weightless utils
void write_weightless(std::ostream& stream, const std::vector<ov::Tensor>& var, const WeightsContext& ctx);
// No allocation needed
void read_weightless(std::istream& stream, std::vector<ov::Tensor>& var, const WeightsContext& ctx);

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

template <typename T, typename H>
void write(std::ostream& stream, const std::unordered_set<T, H>& var) {
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

template <typename T, typename H>
void read(std::istream& stream, std::unordered_set<T, H>& var) {
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
