// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace tp {

/// \brief On-stream layout of a TP compiled blob.
///
/// This is an internal cache artifact, not a portable model format: it embeds
/// one device-plugin blob per rank and only loads back onto a matching device
/// topology. The user always feeds a full, unsharded IR to `compile_model`.
///
///     char[8]   magic = "OVTPBLOB"
///     uint32    version
///     uint32    world_size
///     uint32    strategy
///     uint32    num_sharded_states
///     uint32    state_replicas      (consecutive ranks holding the same slice of the sharded states)
///     [world_size]         device name       : string
///     [num_sharded_states] variable id       : string
///     uint32 num_inputs,  [num_inputs]  port
///     uint32 num_outputs, [num_outputs] port
///     [world_size]         rank blob         : uint64 length + bytes
///
/// A string is a uint32 length + bytes. A port is the model-level port the
/// compiled model reports: node name, element type, partial shape, and
/// uint32 num_names + [num_names] tensor name.
///
/// Everything read back is untrusted input -- a cache file can be stale,
/// truncated or corrupted -- so every length is bounds-checked before it is
/// used to size an allocation or to move the stream.
namespace tp_blob {

inline constexpr char magic[8] = {'O', 'V', 'T', 'P', 'B', 'L', 'O', 'B'};

/// Bump on any layout or semantic change. The version is the only field a
/// reader can act on before it has parsed anything else, so an older blob
/// must never be handed to a newer runtime.
inline constexpr uint32_t version = 3;

/// Device names are short identifiers such as "GPU.0"; anything longer means
/// the stream is not what we think it is.
inline constexpr uint32_t max_device_name_length = 256;

/// Variable ids and tensor names come from the IR and are longer than device
/// names, but still bounded -- a multi-kilobyte id means the stream is not what
/// we think it is.
inline constexpr uint32_t max_variable_id_length = 4096;

/// Bounds on the header counts. Not hardware limits, just the point past
/// which the stream is certainly not a TP blob any more.
inline constexpr uint32_t max_world_size = 64;
inline constexpr uint32_t max_sharded_states = 1u << 16;
inline constexpr uint32_t max_ports = 1u << 16;
inline constexpr uint32_t max_port_names = 1u << 10;

template <class T>
void write_trivial(std::ostream& s, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only trivially copyable types can be written byte-for-byte");
    s.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T read_trivial(std::istream& s) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only trivially copyable types can be read byte-for-byte");
    T v{};
    s.read(reinterpret_cast<char*>(&v), sizeof(T));
    OPENVINO_ASSERT(s.gcount() == static_cast<std::streamsize>(sizeof(T)),
                    "[TP] truncated compiled blob");
    return v;
}

inline void write_string(std::ostream& s, const std::string& v) {
    write_trivial<uint32_t>(s, static_cast<uint32_t>(v.size()));
    s.write(v.data(), static_cast<std::streamsize>(v.size()));
}

inline std::string read_string(std::istream& s, uint32_t max_length = max_device_name_length) {
    const auto length = read_trivial<uint32_t>(s);
    OPENVINO_ASSERT(length <= max_length,
                    "[TP] compiled blob declares a string of ", length,
                    " bytes, which is not plausible; the cache entry is corrupted");
    std::string v(length, '\0');
    s.read(v.data(), static_cast<std::streamsize>(length));
    OPENVINO_ASSERT(s.gcount() == static_cast<std::streamsize>(length),
                    "[TP] truncated compiled blob");
    return v;
}

/// A model-level port as read back from a blob.
struct Port {
    std::string node_name;
    ov::element::Type type;
    ov::PartialShape shape;
    std::unordered_set<std::string> names;
};

inline void write_port(std::ostream& s, const ov::Output<const ov::Node>& port) {
    write_string(s, port.get_node()->get_friendly_name());
    write_string(s, port.get_element_type().to_string());
    write_string(s, port.get_partial_shape().to_string());
    const auto& names = port.get_names();
    write_trivial<uint32_t>(s, static_cast<uint32_t>(names.size()));
    for (const auto& name : names) {
        write_string(s, name);
    }
}

inline Port read_port(std::istream& s) {
    Port port;
    port.node_name = read_string(s, max_variable_id_length);
    port.type = ov::element::Type(read_string(s));
    port.shape = ov::PartialShape(read_string(s, max_variable_id_length));
    const auto num_names = read_trivial<uint32_t>(s);
    OPENVINO_ASSERT(num_names <= max_port_names,
                    "[TP] compiled blob declares a port with ", num_names,
                    " names, which is not plausible; the cache entry is corrupted");
    for (uint32_t i = 0; i < num_names; ++i) {
        port.names.insert(read_string(s, max_variable_id_length));
    }
    return port;
}

}  // namespace tp_blob
}  // namespace tp
}  // namespace ov
