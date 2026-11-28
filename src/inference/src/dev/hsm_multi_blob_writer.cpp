// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_multi_blob_writer.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

namespace {

// Replace with ov::util::MemoryView once it becomes std::span (PR #40243).
using ConstByteSpan = std::span<const std::byte>;

constexpr Header empty_shared_context{
    .magic = BlobMagic::multi,
    .version_major = FormatVersion::major,
    .version_minor = FormatVersion::minor,
    .container_size = sizeof(Header),
    .manifest_offset = sizeof(Header),
    .manifest_size = 0,
};

constexpr BlobMagic magic_of(BlobKind kind) {
    return kind == BlobKind::model ? BlobMagic::single : BlobMagic::multi;
}

bool is_valid_blob(BlobKind kind, const Header& header, size_t size) {
    return header.magic == magic_of(kind) && is_valid_header_fields(header) && header.container_size == size;
}

std::streampos to_streampos(size_t position) {
    return std::streampos(static_cast<std::streamoff>(position));
}

}  // namespace

// Hands one blob's bytes to the writer and keeps a copy of the blob's header to check it at the end.
class MultiBlobWriter::Sink final : public IBlobSink {
public:
    explicit Sink(MultiBlobWriter& writer) : m_writer{writer}, m_start{writer.m_position} {}

    void write(ov::util::MemoryView data) override {
        if (good()) {
            record_header(written_size(), data);
            m_writer.put(data);
        }
    }

    void patch(size_t offset, ov::util::MemoryView data) override {
        OPENVINO_ASSERT(offset <= written_size() && data.size() <= written_size() - offset,
                        "HSM multi-blob writer: patch out of bounds");
        if (good()) {
            record_header(offset, data);
            m_writer.overwrite(m_start + offset, data);
        }
    }

    size_t written_size() const override {
        return m_writer.m_position - m_start;
    }

    size_t origin() const override {
        return m_writer.m_origin + m_start;
    }

    bool good() const override {
        return !m_writer.m_failed;
    }

    bool is_valid(BlobKind kind) const {
        return written_size() >= sizeof(Header) && is_valid_blob(kind, Header::view(m_header.data()), written_size());
    }

private:
    void record_header(size_t offset, ov::util::MemoryView data) {
        if (offset < m_header.size()) {
            const auto count = std::min(data.size(), m_header.size() - offset);
            std::ranges::copy(ConstByteSpan{data.data(), data.size()}.first(count), m_header.begin() + offset);
        }
    }

    MultiBlobWriter& m_writer;
    size_t m_start;  // where this blob starts, as MultiBlobWriter::m_position
    std::array<std::byte, sizeof(Header)> m_header{};
};

MultiBlobWriter::MultiBlobWriter(std::ostream* stream,
                                 std::span<std::byte> buffer,
                                 MultiBlobOpenMode mode,
                                 size_t origin)
    : m_stream{stream},
      m_buffer{buffer},
      m_origin{origin},
      m_position{stream ? static_cast<size_t>(std::streamoff(stream->tellp())) : 0},
      m_failed{false},
      m_needs_shared_context{mode == MultiBlobOpenMode::create} {}

std::optional<MultiBlobWriter> MultiBlobWriter::open(std::ostream& stream, MultiBlobOpenMode mode) {
    if (!stream || stream.tellp() == std::streampos(-1)) {
        return std::nullopt;
    } else {
        return MultiBlobWriter{&stream, {}, mode, 0};
    }
}

std::optional<MultiBlobWriter> MultiBlobWriter::open(std::span<std::byte> dst, MultiBlobOpenMode mode, size_t origin) {
    if (dst.data() == nullptr || dst.size() > std::numeric_limits<size_t>::max() - origin) {
        return std::nullopt;
    } else {
        return MultiBlobWriter{nullptr, dst, mode, origin};
    }
}

bool MultiBlobWriter::good() {
    if (m_stream && !*m_stream) {
        m_failed = true;
    }
    return !m_failed;
}

void MultiBlobWriter::put(ov::util::MemoryView data) {
    if (!good()) {
        return;
    } else if (m_stream) {
        m_stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        m_position += data.size();
        good();
    } else if (data.size() > m_buffer.size() - m_position) {
        m_failed = true;
    } else {
        std::ranges::copy(ConstByteSpan{data.data(), data.size()}, m_buffer.begin() + m_position);
        m_position += data.size();
    }
}

void MultiBlobWriter::overwrite(size_t position, ov::util::MemoryView data) {
    if (!good()) {
        return;
    } else if (m_stream) {
        const auto target = to_streampos(position);
        m_stream->seekp(target);
        m_stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (m_stream->tellp() != target + static_cast<std::streamoff>(data.size())) {
            m_stream->setstate(std::ios::failbit);  // e.g. an append-only stream put the bytes elsewhere
        }
        seek(m_position);
    } else {
        std::ranges::copy(ConstByteSpan{data.data(), data.size()}, m_buffer.begin() + position);
    }
}

void MultiBlobWriter::seek(size_t position) {
    if (m_stream) {
        m_stream->seekp(to_streampos(position));
        good();
    }
}

void MultiBlobWriter::discard(size_t from) noexcept {
    static constexpr std::array<std::byte, 4096> zeros{};
    if (!m_stream) {
        std::ranges::fill(m_buffer.subspan(from, m_position - from), std::byte{0});
        m_position = from;
        return;
    }
    try {
        for (auto at = from; at < m_position && good();) {
            const auto chunk = std::min(m_position - at, zeros.size());
            overwrite(at, {zeros.data(), chunk});
            at += chunk;
        }
        m_position = from;
        seek(from);
    } catch (...) {
        m_failed = true;  // a stream with exceptions enabled failed again while rolling back
    }
}

void MultiBlobWriter::put_shared_context_if_needed(BlobKind kind) {
    if (m_needs_shared_context && kind == BlobKind::model) {
        put(ov::util::as_memory_view(empty_shared_context));
        m_needs_shared_context = false;
    }
}

std::error_code MultiBlobWriter::append(BlobKind kind, ov::util::MemoryView blob) {
    if (!good()) {
        return make_error_code(WriteErrc::write_failed);
    } else if (blob.size() < sizeof(Header) || !is_valid_blob(kind, Header::view(blob.data()), blob.size()) ||
               !ContainerView{blob.data(), blob.size()}.validate()) {
        return make_error_code(WriteErrc::invalid_blob);
    } else {
        put_shared_context_if_needed(kind);
        put(blob);
        m_needs_shared_context = false;
        return good() ? std::error_code{} : make_error_code(WriteErrc::write_failed);
    }
}

std::error_code MultiBlobWriter::append(BlobKind kind, const IBlobEncoder& encoder) {
    put_shared_context_if_needed(kind);
    if (!good()) {
        return make_error_code(WriteErrc::write_failed);
    }

    const auto start = m_position;
    Sink sink{*this};
    try {
        encoder.encode(sink);
    } catch (...) {
        discard(start);
        throw;
    }

    if (!good()) {
        discard(start);
        return make_error_code(WriteErrc::write_failed);
    } else if (!sink.is_valid(kind)) {
        discard(start);
        return make_error_code(good() ? WriteErrc::invalid_blob : WriteErrc::write_failed);
    } else {
        m_needs_shared_context = false;
        return {};
    }
}

}  // namespace v1
}  // namespace ov::runtime::hsm
