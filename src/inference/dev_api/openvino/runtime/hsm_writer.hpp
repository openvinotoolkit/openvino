// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_format.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief Callback handed to #ISectionEncoder::encode() for storing its content. Call it as many times as
 * convenient - each call goes straight to the final destination, so a caller already holding a contiguous
 * source can store it in one call, and a caller producing data incrementally never has to materialize
 * more than one chunk at a time; the writer itself never buffers a whole section on your behalf.
 */
using SectionSink = std::function<void(ov::util::MemoryView data)>;

/**
 * @brief Produces a section's content on demand: call @p sink as many times as convenient. Zero-copy
 * for a fixed-buffer destination (each call writes straight into it); for a stream destination, each call
 * writes straight to the stream too - never a whole-section staging buffer, regardless of size.
 * @note One interface covers both #IWriter::add_section() overloads that take it: given a known size,
 * the calls must sum to exactly that many bytes; without one, the writer measures however many bytes
 * they turn out to be - see that overload's docs for what that implies for the destination.
 */
class ISectionEncoder {
public:
    virtual ~ISectionEncoder() = default;

    /**
     * @brief Writes this section's content to @p sink.
     * @note Runs later, at #IWriter::finalize() time, not when passed to #IWriter::add_section() - any
     * state read here must outlive finalize().
     */
    virtual void encode(const SectionSink& sink) const = 0;
};

/**
 * @brief Owning handle to an #ISectionEncoder. Shared, not exclusive: a caller that already built and
 * holds a long-lived encoder can pass the same instance to multiple #IWriter::add_section() calls
 * (across one writer or several) instead of rebuilding a throwaway one each time.
 */
using SectionEncoderPtr = std::shared_ptr<ISectionEncoder>;

/**
 * @brief Pointer-mode start-offset and reserved-slot-size alignment for a section (ignored for inline-mode).
 * @p offset_align: `0` and `1` are equivalent (no alignment); anything else must be a power of two.
 * @p size_align: `0` (the default) inherits @p offset_align's value - the common case of padding the
 * offset and the slot size the same way; set it explicitly (`1` for no slot padding, or another power of
 * two) only when the slot size genuinely needs to differ from the offset alignment.
 * @note Always relative to the container's own first byte, never to the destination's absolute address -
 * getting that to line up with a real page boundary (e.g. for `mmap`) is the caller's responsibility:
 * place the container's first byte at a page-aligned position in the destination.
 */
struct SectionAlignment {
    size_t offset_align = 1;
    size_t size_align = 0;  //!< 0 = inherit `offset_align`
};

/**
 * @brief Outcome of #IWriter::finalize(), reported via `std::error_code` - a default-constructed
 * (falsy) one means success; compare against #WriteErrc::write_failed for the one failure case.
 */
enum class WriteErrc {
    write_failed = 1,  //!< The destination ran out of room (fixed buffer) or a write failed (stream).
};

/**
 * @brief ADL hook letting a #WriteErrc implicitly convert to/compare against `std::error_code` -
 * `ec.message()` gives the human-readable text.
 */
OPENVINO_RUNTIME_API std::error_code make_error_code(WriteErrc e) noexcept;

class ISectionWriterHandler;

/**
 * @brief Common (device- and destination-agnostic) writer interface for one HSM container.
 * @note #add_section() borrows @p payload (see the concrete class for how long it must stay alive); the
 * #ISectionEncoder overloads produce data on demand instead, for a payload with no existing view (e.g.
 * generated, or only transiently alive). @p device is always explicit - pass #any_device_id for a
 * Core-owned section.
 * @note Tag mode support is asymmetric: an inline @p tag (#SectionTag::is_inline()) is only accepted by
 * the #ov::util::MemoryView overload; both #ISectionEncoder overloads are pointer-mode only and reject
 * an inline @p tag.
 */
class OPENVINO_RUNTIME_API IWriter {
public:
    virtual ~IWriter() = default;

    /**
     * @brief Adds a section owned by @p device.
     *
     * @param device The device that owns the section.
     * @param tag The section tag.
     * @param payload The memory view containing the section's data.
     * @param align The alignment requirements for the section - accepted but ignored for an inline
     * @p tag, so the same value can be reused across inline and pointer-mode calls.
     * @return true if the section was added; false if the destination has run out of room or the write
     * failed.
     */
    virtual bool add_section(DeviceId device,
                             SectionTagReserved tag,
                             ov::util::MemoryView payload,
                             SectionAlignment align = {}) = 0;

    /**
     * @brief Adds a section owned by @p device whose content is produced on demand by @p encoder, for a
     * payload that doesn't exist as a view yet (e.g. generated, or assembled piecemeal) - avoids an
     * intermediate allocation just to call the view overload.
     * @note Pointer-mode only; see the view overload for inline-mode.
     * @note @p encoder is invoked at most once per finalization attempt - see #finalize()'s note on how
     * a repeated call, or a call after an exception, is handled by the concrete implementation. It must
     * write exactly @p size bytes to `sink` (across one or more calls) - never buffer a copy of its own
     * first.
     *
     * @param device The device that owns the section.
     * @param tag The section tag.
     * @param size The size of the section's payload.
     * @param encoder The encoder that produces the section's content.
     * @param align The alignment requirements for the section.
     * @return true if the section was added; false if @p tag is inline, the destination has run out of
     * room, or the write failed.
     */
    virtual bool add_section(DeviceId device,
                             SectionTagReserved tag,
                             size_t size,
                             SectionEncoderPtr encoder,
                             SectionAlignment align = {}) = 0;

    /**
     * @brief Adds a section owned by @p device whose content is produced by @p encoder and whose size is
     * only known once @p encoder has finished writing it (e.g. serializing an object whose encoded length
     * depends on its runtime data).
     * @note @p encoder is invoked at most once per finalization attempt - see #finalize()'s note on how
     * a repeated call, or a call after an exception, is handled by the concrete implementation.
     * Pointer-mode only (same restriction as the sized overload for inline tags). Forces #finalize() to
     * revisit the destination once every real size is known, instead of computing the header up front -
     * a stream destination must be seekable if any section uses this overload.
     *
     * @param device The device that owns the section.
     * @param tag The section tag.
     * @param encoder The encoder that produces the section's content.
     * @param align The alignment requirements for the section.
     * @return true if the section was added; false if @p tag is inline, the destination has run out of
     * room, or the write failed.
     */
    virtual bool add_section(DeviceId device,
                             SectionTagReserved tag,
                             SectionEncoderPtr encoder,
                             SectionAlignment align = {}) = 0;

    /**
     * @brief Writes the header, every section, and the manifest.
     * @note Call after the last add_section(). Behavior of a repeated call, and of a call after an
     * exception escapes a previous one, is defined by the concrete implementation.
     */
    virtual std::error_code finalize() = 0;

    /**
     * @brief Lets each of @p handlers contribute its own sections, in order, via #ISectionWriterHandler::
     * handle_section(). Symmetric to #Reader::read_sections() on the read side.
     * @param handlers The list of section writer handlers to invoke.
     */
    void add_sections(const std::vector<ISectionWriterHandler*>& handlers);
};

/**
 * @brief Writer-side plugin hook: a plugin contributes its own proprietary sections to a container.
 * Symmetric to #ISectionReaderHandler - #IWriter::add_sections() is the driver, taking a list of
 * handlers, exactly like #Reader::read_sections() does.
 */
class OPENVINO_RUNTIME_API ISectionWriterHandler {
public:
    virtual ~ISectionWriterHandler() = default;

    /**
     * @brief Reserves this handler's sections on @p writer, via any #IWriter::add_section() overload.
     * @note Called by #IWriter::add_sections(), not directly - see its docs.
     */
    virtual void handle_section(IWriter& writer) const = 0;
};

}  // namespace v1
}  // namespace ov::runtime::hsm

template <>
struct std::is_error_code_enum<ov::runtime::hsm::WriteErrc> : std::true_type {};
