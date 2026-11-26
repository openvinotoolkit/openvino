// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <system_error>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/// What a blob holds; decides the magic it must carry.
enum class BlobKind : uint8_t {
    shared_context,  //!< #BlobMagic::multi
    model,           //!< #BlobMagic::single
};

/// How MultiBlobWriter::open() treats the destination.
enum class MultiBlobOpenMode {
    /// The destination has no blobs yet. If the first blob added is a model, an empty shared-context blob is
    /// written before it.
    create,
    /// The destination already holds a valid multi-blob file. New blobs follow it, with no rule about what comes
    /// first. The writer can't check the existing content - the caller is responsible for it.
    append,
};

/**
 * @brief Writes HSM blobs one after another into a single file, as read by #MultiBlobView. It doesn't look
 * inside the blobs; it only checks that each one is a valid container of the expected #BlobKind.
 * @note Each append() writes immediately. An invalid blob returns WriteErrc::invalid_blob and leaves the
 * writer usable; a failed write returns WriteErrc::write_failed, and so does every later append().
 */
class OPENVINO_RUNTIME_API MultiBlobWriter final {
public:
    /**
     * @brief Opens a writer on @p stream.
     * @param stream Must stay valid while the writer is used, and be seekable (needed to patch headers and
     * to roll back an invalid blob). Writing starts at its current position, which is also the first blob's
     * #IBlobSink::origin().
     * @param mode See #MultiBlobOpenMode.
     * @return std::nullopt if @p stream is not good or can't report its position.
     */
    static std::optional<MultiBlobWriter> open(std::ostream& stream,
                                               MultiBlobOpenMode mode = MultiBlobOpenMode::create);

    /**
     * @brief Opens a writer on a caller-owned buffer; writing starts at `dst[0]`.
     * @param dst Destination buffer; must stay valid while the writer is used. To add blobs after existing
     * ones, pass the free part only (e.g. `whole.subspan(used)`) with MultiBlobOpenMode::append.
     * @param mode See #MultiBlobOpenMode.
     * @param origin Position of `dst[0]` in the whole file. Used only to align sections (see
     * #IBlobSink::origin()); it doesn't move where writing starts.
     * @return std::nullopt if @p dst has no data or `origin + dst.size()` overflows.
     */
    static std::optional<MultiBlobWriter> open(std::span<std::byte> dst,
                                               MultiBlobOpenMode mode = MultiBlobOpenMode::create,
                                               size_t origin = 0);

    /// @note Move-only: two copies would write to the same destination from diverging positions.
    MultiBlobWriter(const MultiBlobWriter&) = delete;
    MultiBlobWriter& operator=(const MultiBlobWriter&) = delete;
    MultiBlobWriter(MultiBlobWriter&&) noexcept = default;
    MultiBlobWriter& operator=(MultiBlobWriter&&) noexcept = default;

    /**
     * @brief Writes @p blob, after checking that it is a valid container of @p kind; writes nothing if it
     * isn't.
     * @return Empty on success, WriteErrc::invalid_blob or WriteErrc::write_failed otherwise.
     */
    [[nodiscard]] std::error_code append(BlobKind kind, ov::util::MemoryView blob);

    /**
     * @brief Runs @p encoder now, writing its output, then checks it is a valid container of @p kind.
     * If it isn't, or @p encoder throws, the written bytes are zero-filled and the write position moves
     * back to where the blob started; an exception is then rethrown.
     * @note A stream can't be shortened, so after a rollback the zeros stay in it until later blobs
     * overwrite them; #MultiBlobView stops reading there.
     * @return Empty on success, WriteErrc::invalid_blob or WriteErrc::write_failed otherwise.
     */
    [[nodiscard]] std::error_code append(BlobKind kind, const IBlobEncoder& encoder);

private:
    class Sink;

    MultiBlobWriter(std::ostream* stream, std::span<std::byte> buffer, MultiBlobOpenMode mode, size_t origin);

    bool good();
    void put(ov::util::MemoryView data);
    void overwrite(size_t position, ov::util::MemoryView data);
    void seek(size_t position);
    void discard(size_t from) noexcept;
    void put_shared_context_if_needed(BlobKind kind);

    std::ostream* m_stream;         // null for a buffer destination
    std::span<std::byte> m_buffer;  // unused for a stream destination
    size_t m_origin;                // file position of m_buffer[0]; 0 for a stream, whose positions already are
    size_t m_position;              // next write position: in the stream, or an index into m_buffer
    bool m_failed;                  // a write has failed (sticky)
    bool m_needs_shared_context;    // create mode, and no blob has been written yet
};

}  // namespace v1
}  // namespace ov::runtime::hsm
