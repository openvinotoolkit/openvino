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
#include "orc.hpp"
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

const constexpr char* NPUW_SERIALIZATION_VERSION = "0.24";

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

// --------------------------------------------------------------------------
// s11n: context types, helper aliases, and utilities
// --------------------------------------------------------------------------
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

    // NOTE: This constructor should only be used when exporting blobs.
    WeightsContext(bool _is_weightless, const std::unordered_map<const void*, std::size_t>& _const_to_offset);

    // NOTE: This constructor is used on blob import to carry the resolved weight source
    // (embedded weights, mmap'ed weights file, or model-backed constants cache).
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

    SubmodelDeserializeCtx(const std::shared_ptr<const ov::IPlugin>& _plugin,
                           const ov::SoPtr<ov::ICompiledModel>& _compiled_model,
                           std::function<std::string(std::size_t)> _device_by_index,
                           std::function<std::map<std::string, Any>(const std::string&)> _import_config_for_device)
        : plugin(_plugin),
          compiled_model(_compiled_model),
          device_by_index(std::move(_device_by_index)),
          import_config_for_device(std::move(_import_config_for_device)) {}

    std::shared_ptr<const ov::IPlugin> plugin;
    std::string device;
    const ov::SoPtr<ov::ICompiledModel>& compiled_model;
    std::map<std::string, Any> import_config;
    std::function<std::string(std::size_t)> device_by_index;
    std::function<std::map<std::string, Any>(const std::string&)> import_config_for_device;
};

BF16Cache get_bf16_consts(const std::shared_ptr<ov::Model>& model);

// The NPUW stream type IS the ORC stream.
using Stream = ::ov::npuw::orc::Stream;

// Bring all orc serialize overloads into s11n namespace so that
// qualified ov::npuw::s11n::serialize(...) call sites continue to work.
using ::ov::npuw::orc::serialize;

using TensorAllocator = std::function<ov::Tensor(const ov::element::Type&, const ov::Shape&)>;

template <typename T>
void write(std::ostream& stream, const T& var) {
    auto stream_io = Stream::writer(stream);
    auto& mutable_var = const_cast<std::remove_const_t<T>&>(var);
    stream_io & mutable_var;
}

template <typename T>
void read(std::istream& stream, T& var) {
    auto stream_io = Stream::reader(stream);
    stream_io & var;
}

inline void write_any(std::ostream& stream, const ov::Any& var) {
    write(stream, var);
}

inline void read_any(std::istream& stream, ov::Any& var) {
    read(stream, var);
}

}  // namespace s11n

// --------------------------------------------------------------------------
// NPUW-specific orc serializers (in orc namespace for ADL from orc::Stream)
// --------------------------------------------------------------------------
namespace orc {

// Raw stream position — used by the compiled model blob header.
void serialize(Stream& stream, std::streampos& var);

// NPUW compiled structures
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

// OpenVINO runtime types
void serialize(Stream& stream, ov::Tensor& var);
void serialize(Stream& stream, ov::CacheMode& var);
void serialize(Stream& stream, ov::element::Type& var);
void serialize(Stream& stream, ov::hint::PerformanceMode& var);
void serialize(Stream& stream, ov::Any& var);
void serialize(Stream& stream, ov::AnyMap& var);
void serialize(Stream& stream, ov::Output<const ov::Node>& var);
void serialize(Stream& stream, std::shared_ptr<ov::op::v0::Parameter>& var);
void serialize(Stream& stream, std::shared_ptr<ov::Node>& var);

// NPU plugin configuration
void serialize(Stream& stream, ::intel_npu::Config& var);

// NPUW weight types
void serialize(Stream& stream, ov::npuw::weights::LazyTensor& var);
void serialize(Stream& stream, ov::npuw::weights::Bank& var);

// Weightless tensor utilities
void serialize_weightless(Stream& stream, std::vector<ov::Tensor>& var, const s11n::WeightsContext& ctx);
void transfer_tensor(Stream& stream, ov::Tensor& var, const s11n::TensorAllocator& allocator = {});

}  // namespace orc

// Reopen s11n for weightless helpers that need the orc declarations above.
namespace s11n {

inline void write_weightless(std::ostream& stream, const std::vector<ov::Tensor>& var, const WeightsContext& ctx) {
    auto stream_io = Stream::writer(stream);
    auto& mutable_var = const_cast<std::vector<ov::Tensor>&>(var);
    orc::serialize_weightless(stream_io, mutable_var, ctx);
}

inline void read_weightless(std::istream& stream, std::vector<ov::Tensor>& var, const WeightsContext& ctx) {
    auto stream_io = Stream::reader(stream);
    orc::serialize_weightless(stream_io, var, ctx);
}

}  // namespace s11n

}  // namespace npuw
}  // namespace ov
