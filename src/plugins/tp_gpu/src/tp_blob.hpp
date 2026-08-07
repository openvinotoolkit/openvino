// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>

#include "openvino/core/except.hpp"

namespace ov {
namespace tp_gpu {

/// \brief On-stream layout of a TP_GPU compiled blob.
///
/// This is an internal cache artifact, not a portable model format: it embeds
/// one intel_gpu blob per rank and only loads back onto a matching device
/// topology.  The user always feeds a full, unsharded IR to `compile_model`.
///
///     char[8]   magic = "OVTPGPU"
///     uint32    version
///     uint32    world_size
///     uint32    num_collectives
///     [world_size] device name : uint32 length + bytes
///     [world_size] rank blob   : uint64 length + bytes
///
/// Everything read back is untrusted input -- a cache file can be stale,
/// truncated or corrupted -- so every length is bounds-checked before it is
/// used to size an allocation or to move the stream.
namespace tp_blob {

inline constexpr char magic[8] = {'O', 'V', 'T', 'P', 'G', 'P', 'U', '\0'};

/// Bump on any layout or semantic change.  The version is the only field a
/// reader can act on before it has parsed anything else, so an older blob
/// must never be handed to a newer runtime.
inline constexpr uint32_t version = 1;

/// Device names are short identifiers such as "GPU.0"; anything longer means
/// the stream is not what we think it is.
inline constexpr uint32_t max_device_name_length = 256;

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
                    "[TP_GPU] truncated compiled blob");
    return v;
}

inline void write_string(std::ostream& s, const std::string& v) {
    write_trivial<uint32_t>(s, static_cast<uint32_t>(v.size()));
    s.write(v.data(), static_cast<std::streamsize>(v.size()));
}

inline std::string read_string(std::istream& s) {
    const auto length = read_trivial<uint32_t>(s);
    OPENVINO_ASSERT(length <= max_device_name_length,
                    "[TP_GPU] compiled blob declares a device name of ", length,
                    " bytes, which is not plausible; the cache entry is corrupted");
    std::string v(length, '\0');
    s.read(v.data(), static_cast<std::streamsize>(length));
    OPENVINO_ASSERT(s.gcount() == static_cast<std::streamsize>(length),
                    "[TP_GPU] truncated compiled blob");
    return v;
}

}  // namespace tp_blob
}  // namespace tp_gpu
}  // namespace ov
