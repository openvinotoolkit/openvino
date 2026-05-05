// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "attention.hpp"
#include "host_flash_attention.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/file_handle.hpp"
#include "pyramid_attention.hpp"
#include "spatial.hpp"

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

const constexpr char* NPUW_SERIALIZATION_VERSION = "0.23";

// Forward declaration
namespace intel_npu {
class Config;
}  // namespace intel_npu

namespace ov {

// Forward declaration
class Any;
class Node;
class Tensor;
class IPlugin;
class ICompiledModel;
template <class>
class Output;
template <class>
class SharedBuffer;
template <class>
struct SoPtr;
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
struct PyramidAttention;
struct HostFlashAttention;
struct MoEExperts;
struct MoEDownstream;
}  // namespace compiled
namespace weights {
class LazyTensor;
class Bank;
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
                   const BF16Cache& _bf16_consts,
                   const ov::FileHandleProvider& _handle_provider = nullptr);

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
    ov::FileHandleProvider handle_provider = nullptr;
};

// Context for deserializing submodels with dynamic attention mechanisms
// (Pyramid Attention, Host Flash Attention, etc.)
// Provides plugin, device, and compiled model reference for proper deserialization
struct SubmodelDeserializeCtx {
    SubmodelDeserializeCtx(const std::shared_ptr<const ov::IPlugin>& _plugin,
                           const std::string& _device,
                           const ov::SoPtr<ov::ICompiledModel>& _compiled_model,
                           const std::map<std::string, Any>& _import_config = {})
        : plugin(_plugin),
          device(_device),
          compiled_model(_compiled_model),
          import_config(_import_config) {}

    std::shared_ptr<const ov::IPlugin> plugin;
    std::string device;
    const ov::SoPtr<ov::ICompiledModel>& compiled_model;
    std::map<std::string, Any> import_config;
};

BF16Cache get_bf16_consts(const std::shared_ptr<ov::Model>& model);

class Stream {
public:
    static Stream reader(std::istream& stream) {
        return Stream(&stream, nullptr);
    }

    static Stream writer(std::ostream& stream) {
        return Stream(nullptr, &stream);
    }

    bool input() const {
        return m_input != nullptr;
    }

    bool output() const {
        return m_output != nullptr;
    }

    template <typename T>
    Stream& operator&(T&& value) {
        using plain_type = std::remove_const_t<std::remove_reference_t<T>>;
        serialize(*this, const_cast<plain_type&>(value));
        return *this;
    }

    void operator()() {}

    template <typename T, typename... Ts>
    void operator()(T&& value, Ts&&... values) {
        (*this) & std::forward<T>(value);
        (*this)(std::forward<Ts>(values)...);
    }

    void bytes(void* data, std::size_t size) {
        const auto stream_size = to_streamsize(size);
        if (output()) {
            m_output->write(reinterpret_cast<const char*>(data), stream_size);
        } else {
            m_input->read(reinterpret_cast<char*>(data), stream_size);
        }
    }

private:
    static std::streamsize to_streamsize(std::size_t size) {
        if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::overflow_error("Stream::bytes() size exceeds std::streamsize range");
        }
        return static_cast<std::streamsize>(size);
    }

    Stream(std::istream* input, std::ostream* output) : m_input(input), m_output(output) {}

    std::istream* m_input = nullptr;
    std::ostream* m_output = nullptr;
};

void serialize(Stream& stream, std::streampos& var);
void serialize(Stream& stream, std::string& var);
void serialize(Stream& stream, bool& var);
void serialize(Stream& stream, float& var);
void serialize(Stream& stream, ov::npuw::compiled::Spatial& var);
void serialize(Stream& stream, ov::npuw::compiled::Spatial::Param& var);
void serialize(Stream& stream, ov::npuw::compiled::Attention& var);
void serialize(Stream& stream, ov::npuw::compiled::Attention::Param& var);
void serialize(Stream& stream, ov::npuw::compiled::PyramidAttention& var);
void serialize(Stream& stream, ov::npuw::compiled::PyramidAttentionInfo& var);
void serialize(Stream& stream, ov::npuw::compiled::PyramidAttentionInfo::Param& var);
void serialize(Stream& stream, ov::npuw::compiled::HostFlashAttention& var);
void serialize(Stream& stream, ov::npuw::compiled::MoEExperts& var);
void serialize(Stream& stream, ov::npuw::compiled::MoEDownstream& var);
void serialize(Stream& stream, ov::Tensor& var);
void serialize(Stream& stream, ::intel_npu::Config& var);
void serialize(Stream& stream, ov::Any& var);
void serialize(Stream& stream, ov::npuw::weights::LazyTensor& var);
void serialize(Stream& stream, ov::npuw::weights::Bank& var);
void serialize(Stream& stream, ov::CacheMode& var);
void serialize(Stream& stream, ov::element::Type& var);
void serialize(Stream& stream, ov::hint::PerformanceMode& var);
void serialize(Stream& stream, std::map<std::string, Any>& var);
void serialize(Stream& stream, ov::Output<const ov::Node>& var);
void serialize(Stream& stream, std::shared_ptr<ov::op::v0::Parameter>& var);
void serialize(Stream& stream, std::shared_ptr<ov::Node>& var);

// Weightless utils
void serialize_weightless(Stream& stream, std::vector<ov::Tensor>& var, const WeightsContext& ctx);

// Forward declaration
template <typename T1, typename T2>
void serialize(Stream& stream, std::pair<T1, T2>& var);
template <typename T>
void serialize(Stream& stream, std::vector<T>& var);
template <typename T, size_t N>
void serialize(Stream& stream, std::array<T, N>& var);

template <typename T>
void write(std::ostream& stream, const T& var) {
    auto stream_io = Stream::writer(stream);
    auto& mutable_var = const_cast<std::remove_const_t<T>&>(var);
    serialize(stream_io, mutable_var);
}

template <typename T>
void read(std::istream& stream, T& var) {
    auto stream_io = Stream::reader(stream);
    serialize(stream_io, var);
}

inline void write_any(std::ostream& stream, const ov::Any& var) {
    write(stream, var);
}

inline void read_any(std::istream& stream, ov::Any& var) {
    read(stream, var);
}

inline void write_weightless(std::ostream& stream, const std::vector<ov::Tensor>& var, const WeightsContext& ctx) {
    auto stream_io = Stream::writer(stream);
    auto& mutable_var = const_cast<std::vector<ov::Tensor>&>(var);
    serialize_weightless(stream_io, mutable_var, ctx);
}

inline void read_weightless(std::istream& stream, std::vector<ov::Tensor>& var, const WeightsContext& ctx) {
    auto stream_io = Stream::reader(stream);
    serialize_weightless(stream_io, var, ctx);
}

// Serialization
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void serialize(Stream& stream, T& var) {
    stream.bytes(&var, sizeof var);
}

template <typename T1, typename T2>
void serialize(Stream& stream, std::pair<T1, T2>& var) {
    stream & var.first & var.second;
}

template <typename T>
void serialize(Stream& stream, std::vector<T>& var) {
    if (stream.output()) {
        auto size = var.size();
        stream & size;
        for (std::size_t i = 0; i < var.size(); ++i) {
            if constexpr (std::is_same_v<T, bool>) {
                bool el = var[i];
                stream & el;
            } else {
                auto& el = var[i];
                stream & el;
            }
        }
    } else {
        var.clear();
        std::size_t var_size = 0;
        stream & var_size;
        var.reserve(var_size);
        for (std::size_t i = 0; i < var_size; ++i) {
            T elem;
            stream & elem;
            var.push_back(std::move(elem));
        }
    }
}

template <typename T, size_t N>
void serialize(Stream& stream, std::array<T, N>& var) {
    for (auto& el : var) {
        stream & el;
    }
}

template <typename T>
void serialize(Stream& stream, std::unordered_set<T>& var) {
    if (stream.output()) {
        auto size = var.size();
        stream & size;
        for (const auto& el : var) {
            auto value = el;
            stream & value;
        }
    } else {
        var.clear();
        std::size_t var_size = 0;
        stream & var_size;
        for (std::size_t i = 0; i < var_size; ++i) {
            T elem;
            stream & elem;
            var.insert(std::move(elem));
        }
    }
}

template <typename T, typename H>
void serialize(Stream& stream, std::unordered_set<T, H>& var) {
    if (stream.output()) {
        auto size = var.size();
        stream & size;
        for (const auto& el : var) {
            auto value = el;
            stream & value;
        }
    } else {
        var.clear();
        std::size_t var_size = 0;
        stream & var_size;
        for (std::size_t i = 0; i < var_size; ++i) {
            T elem;
            stream & elem;
            var.insert(std::move(elem));
        }
    }
}

template <typename K, typename V>
void serialize(Stream& stream, std::map<K, V>& var) {
    if (stream.output()) {
        auto size = var.size();
        stream & size;
        for (auto& el : var) {
            auto pair = el;
            stream & pair;
        }
    } else {
        var.clear();
        std::size_t var_size = 0;
        stream & var_size;
        for (std::size_t i = 0; i < var_size; ++i) {
            std::pair<K, V> elem;
            stream & elem;
            var[elem.first] = elem.second;
        }
    }
}

template <typename T>
void serialize(Stream& stream, std::optional<T>& var) {
    bool has_value = var.has_value();
    stream & has_value;
    if (has_value) {
        if (stream.output()) {
            auto& value = var.value();
            stream & value;
        } else {
            T value;
            stream & value;
            var = std::move(value);
        }
    } else {
        var.reset();
    }
}

using TensorAllocator = std::function<ov::Tensor(const ov::element::Type&, const ov::Shape&)>;
void transfer_tensor(Stream& stream, ov::Tensor& var, const TensorAllocator& allocator = {});

}  // namespace s11n
}  // namespace npuw
}  // namespace ov
