// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <system_error>
#include <variant>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/**
 * @brief Writer that accumulates sections as lightweight descriptors and writes them all at #finalize() -
 * #add_section() borrows @p payload (keep it alive until then); for a transient source, use an
 * #ISectionEncoder instead (see #add_section()'s encoder overload) - the writer itself never holds an
 * owned payload buffer.
 * @note When every section's size is known up front, the header is computed in one pure pass and written
 * first - no seeking back. Using the unsized overload of #add_section() (size only known after encoding)
 * forces the destination to be revisited once, after every real size is known - see #open(std::ostream&).
 */
class OPENVINO_RUNTIME_API DeferredWriter final : public IWriter {
public:
    /**
     * @brief Opens a DeferredWriter that writes into the given output stream.
     *
     * @param stream The output stream to write into. Must remain valid until #finalize() is called, and
     * be seekable - needed both to retry after a thrown exception and to revisit the destination when
     * the unsized overload of #add_section() is used.
     * @return A DeferredWriter instance if the stream is usable; std::nullopt otherwise.
     */
    static std::optional<DeferredWriter> open(std::ostream& stream);

    /**
     * @brief Opens a DeferredWriter that writes into the given preallocated buffer.
     *
     * @param dst The preallocated buffer to write into. Must remain valid until #finalize() is called.
     * @param size The size of the preallocated buffer in bytes.
     * @return A DeferredWriter instance if the buffer is usable; std::nullopt otherwise.
     */
    static std::optional<DeferredWriter> open(std::byte* dst, size_t size);

    /// @note Move-only: copying would silently duplicate owned/generated section payloads.
    DeferredWriter(const DeferredWriter&) = delete;
    DeferredWriter& operator=(const DeferredWriter&) = delete;
    DeferredWriter(DeferredWriter&&) noexcept = default;
    DeferredWriter& operator=(DeferredWriter&&) noexcept = default;

    bool add_section(DeviceId device,
                     SectionTagReserved tag,
                     ov::util::MemoryView payload,
                     SectionAlignment align = {}) override;
    bool add_section(DeviceId device,
                     SectionTagReserved tag,
                     size_t size,
                     SectionEncoderPtr encoder,
                     SectionAlignment align = {}) override;
    bool add_section(DeviceId device,
                     SectionTagReserved tag,
                     SectionEncoderPtr encoder,
                     SectionAlignment align = {}) override;

    /**
     * @brief See #IWriter::finalize().
     *
     * @return A repeated call after success, or after a normal (non-throwing) failure, just reports the
     * same outcome again. A repeated call after a thrown exception instead retries the whole attempt
     * from the container's start - each #ISectionEncoder may run again.
     */
    std::error_code finalize() override;

private:
    explicit DeferredWriter(std::ostream& stream) noexcept;
    DeferredWriter(std::byte* dst, size_t capacity) noexcept;

    // A section produced on demand at finalize() time - its size is known up front.
    struct PendingEncode {
        size_t size;
        SectionEncoderPtr encoder;
    };

    // One queued, not-yet-written section; `align` is already resolved to concrete powers of two. Field
    // order (largest-alignment-first) keeps padding to just the trailing (device, tag) pair. A bare
    // ISectionEncoder alternative means the unsized overload was used - size discovered only after it runs.
    struct PendingSection {
        SectionAlignment align;
        std::variant<ov::util::MemoryView, PendingEncode, SectionEncoderPtr> payload;
        DeviceId device;
        SectionTagReserved tag;

        PendingSection(SectionAlignment align, ov::util::MemoryView payload, DeviceId device, SectionTagReserved tag);
        PendingSection(SectionAlignment align, PendingEncode payload, DeviceId device, SectionTagReserved tag);
        PendingSection(SectionAlignment align, SectionEncoderPtr payload, DeviceId device, SectionTagReserved tag);
    };

    // Where bytes actually go - a stream (own running size tracked, since a stream has no addressable
    // pointer to fill directly) or a caller-owned fixed buffer (additionally tracks capacity and a
    // sticky failure flag once it runs out of room). A raw pointer, not a reference, so this stays
    // assignable when held in m_destination below.
    struct StreamDestination {
        std::ostream* stream = nullptr;  // destination stream
        std::streampos start{};          // container's first byte, captured once at open()
        size_t size = 0;                 // bytes written so far (the write cursor)
    };
    struct BufferDestination {
        std::byte* dst = nullptr;  // destination buffer
        size_t capacity = 0;       // total usable bytes in dst
        size_t size = 0;           // bytes written so far (the write cursor)
        bool good = true;          // false once a write has failed (sticky)
    };

    static std::optional<size_t> reserve(BufferDestination& destination, size_t extra);

    void write(ov::util::MemoryView data);
    void write_zeros(size_t count);
    void patch(size_t offset, ov::util::MemoryView data);
    size_t written_size() const;
    bool destination_good() const;
    void reset_destination();
    void fail_destination();

    static void write_into(StreamDestination& destination, ov::util::MemoryView data);
    static void write_into(BufferDestination& destination, ov::util::MemoryView data);
    static void patch_into(StreamDestination& destination, size_t offset, ov::util::MemoryView data);
    static void patch_into(BufferDestination& destination, size_t offset, ov::util::MemoryView data);
    static bool is_good(const StreamDestination& destination);
    static bool is_good(const BufferDestination& destination);

    ManifestEntry write_section(DeviceId device,
                                SectionTagReserved tag,
                                SectionAlignment alignment,
                                ov::util::MemoryView payload);
    ManifestEntry write_section(DeviceId device,
                                SectionTagReserved tag,
                                SectionAlignment alignment,
                                const PendingEncode& payload);
    ManifestEntry write_section(DeviceId device,
                                SectionTagReserved tag,
                                SectionAlignment alignment,
                                const SectionEncoderPtr& encoder);

    std::variant<StreamDestination, BufferDestination> m_destination;
    std::vector<PendingSection> m_sections;
    std::optional<std::error_code> m_result;
    bool m_has_unsized_section;
};

}  // namespace v1
}  // namespace ov::runtime::hsm
