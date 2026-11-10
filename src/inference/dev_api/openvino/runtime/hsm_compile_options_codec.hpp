// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Plugin-agnostic codec for the #compile_options_tag section payload.
 * @file openvino/runtime/hsm_compile_options_codec.hpp
 *
 * The payload is one encoded `ov::AnyMap`, written and parsed as a stream - no lengths, no size limits:
 * @verbatim
   Payload  := Entry* FS digest
   Entry    := Leaf | Map | Reserved
   Leaf     := key US value US
   Map      := key GS Entry* GS
   Reserved := key RS content RS
   @endverbatim
 * - Separators: `US` = 0x1F, `GS` = 0x1D, `RS` = 0x1E, `FS` = 0x1C. `key` and `value` are byte strings
 *   containing none of `NUL`, `US`, `GS`, `RS`, `FS`; `content` is any bytes except `RS` and `FS`.
 * - A Leaf holds a string value, a Map a nested options map. Entries of one map are unique by key, whatever
 *   their kind; they may come in any order.
 * - An entry with an empty key is skipped by the decoder. A `GS` with no key bytes before it closes the current
 *   map instead.
 * - Reserved is for future extensions: the decoder skips it entirely, its key included; the encoder never writes it.
 * - `digest` is 8 bytes in native byte order: `h = 0`, then `h = ov::util::u64_hash_combine(h, byte)` for every
 *   payload byte before it, `FS` included. It detects accidental corruption only. The section size must be exactly
 *   the position of `FS` + 1 + 8.
 */

#pragma once

#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief #ISectionEncoder for a #compile_options_tag section: borrows @p options and writes them straight to the
 * sink, with no intermediate buffer - for the unsized #IWriter::add_section() overload.
 * @note The borrowed map must outlive #IWriter::finalize().
 * @note Value types:
 * - `std::string` is written as is;
 * - `ov::AnyMap` and `std::map<std::string, ov::AnyMap>` (e.g. `ov::device::properties`) become nested maps;
 * - any other value is written as its `ov::Any::print()` text in the classic "C" locale, floating-point values
 *   with `max_digits10` digits.
 * Such a value must read back unchanged through `ov::util::Read` - callbacks, pointers and tensors can't be
 * stored, leave them out of @p options.
 * @note Every nested map is decoded as `ov::AnyMap` and every leaf as `std::string`, whatever type was encoded.
 */
class OPENVINO_RUNTIME_API CompileOptionsEncoder final : public ISectionEncoder {
public:
    explicit CompileOptionsEncoder(const ov::AnyMap& options) noexcept : m_options{options} {}
    explicit CompileOptionsEncoder(ov::AnyMap&&) = delete;
    explicit CompileOptionsEncoder(const ov::AnyMap&&) = delete;

    /**
     * @throw ov::Exception if a key is empty, an `ov::Any` is empty, a key or value contains a reserved byte
     * (`NUL`, `US`, `GS`, `RS`, `FS`), a non-string value has no text form, or its conversion throws. Part of the
     * section may already be written to @p sink by then.
     */
    void encode(const SectionSink& sink) const override;

private:
    const ov::AnyMap& m_options;
};

/**
 * @brief Decodes a #compile_options_tag payload.
 * @param section Section to decode; its tag must be #compile_options_tag.
 * @return Decoded options; `std::nullopt` if the section is malformed, the hash does not match, or a key appears
 * more than once within the same map level (as a leaf, a nested map, or both).
 */
OPENVINO_RUNTIME_API std::optional<ov::AnyMap> decode_compile_options(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
