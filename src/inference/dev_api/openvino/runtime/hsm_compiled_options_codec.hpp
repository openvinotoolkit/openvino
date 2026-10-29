// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Plugin-agnostic codec for the #compiled_options_tag section payload.
 * @file openvino/runtime/hsm_compiled_options_codec.hpp
 *
 * The payload is one encoded `ov::AnyMap` (native byte order, like the rest of HSM):
 * @verbatim
   Map   := u32 count, Entry[count]
   Entry := u32 key_len, key bytes, u8 kind, u32 value_len, value bytes
   @endverbatim
 * Value kinds (#CompiledOptionKind): a string (the value's `ov::Any` string form), a nested Map, or an empty
 * `ov::Any` (`value_len == 0`). A reader skips an entry of an unknown kind using `value_len`.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/// Maximum size of the whole #compiled_options_tag payload, in bytes.
inline constexpr size_t max_compiled_options_size = 1024 * 1024;

/// Maximum size of a single key and of a single string value, in bytes.
inline constexpr size_t max_compiled_option_value_size = 64 * 1024;

/// Maximum number of nested map levels, the top-level map included.
inline constexpr size_t max_compiled_options_depth = 8;

/// Wire value of an entry's `kind` field - never change or reuse a value once shipped.
enum class CompiledOptionKind : uint8_t {
    string = 0,  //!< Value bytes are the option's `ov::Any` string form.
    map = 1,     //!< Value bytes are a nested Map.
    empty = 2,   //!< Empty `ov::Any`; `value_len` is always 0.
};

/**
 * @brief Encodes @p options into a #compiled_options_tag payload.
 * @note A value holding an `ov::AnyMap` is encoded as a nested map; any other non-empty value as its
 * `ov::Any::as<std::string>()` form. A type without a string conversion yields an empty string - leave
 * such values (callbacks, tensors, model pointers) out of @p options.
 * @note Pass the result to the #ov::util::MemoryView overload of #IWriter::add_section(), keeping it alive until
 * #IWriter::finalize().
 * @param options Options to encode.
 * @return Encoded payload.
 * @throw ov::Exception if a key is empty, a key or string value exceeds #max_compiled_option_value_size,
 * maps nest deeper than #max_compiled_options_depth, the payload exceeds #max_compiled_options_size, or a
 * value's string conversion throws.
 */
OPENVINO_RUNTIME_API std::vector<std::byte> encode_compiled_options(const ov::AnyMap& options);

/**
 * @brief Lazy #ISectionEncoder for a #compiled_options_tag section: borrows @p options and encodes them only
 * in #encode(), via #encode_compiled_options() - for the unsized #IWriter::add_section() overload.
 * @note The borrowed map must outlive #IWriter::finalize(); #encode() throws what #encode_compiled_options() does.
 */
class CompiledOptionsEncoder final : public ISectionEncoder {
public:
    explicit CompiledOptionsEncoder(const ov::AnyMap& options) noexcept : m_options(options) {}
    explicit CompiledOptionsEncoder(ov::AnyMap&&) = delete;
    explicit CompiledOptionsEncoder(const ov::AnyMap&&) = delete;

    void encode(const SectionSink& sink) const override {
        const auto bytes = encode_compiled_options(m_options);
        sink({bytes.data(), bytes.size()});
    }

private:
    const ov::AnyMap& m_options;
};

/**
 * @brief Decodes a #compiled_options_tag payload.
 * @note Every count and length is checked against the bytes remaining and the `max_compiled_*` limits before
 * anything is allocated, so a corrupted or hostile payload cannot trigger a large allocation.
 * @param payload Encoded payload.
 * @return Decoded options, with string values as `std::string`; `std::nullopt` if the payload is truncated,
 * has trailing bytes, an empty or duplicate key, exceeds a limit, or a nested map doesn't span exactly its
 * `value_len`.
 */
OPENVINO_RUNTIME_API std::optional<ov::AnyMap> decode_compiled_options(ov::util::MemoryView payload) noexcept;

/**
 * @overload Same as #decode_compiled_options(ov::util::MemoryView), for @p section backed by a buffer or a
 * stream; a section larger than #max_compiled_options_size is rejected before it is read.
 * @param section Section to decode; its tag must be #compiled_options_tag.
 */
OPENVINO_RUNTIME_API std::optional<ov::AnyMap> decode_compiled_options(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
