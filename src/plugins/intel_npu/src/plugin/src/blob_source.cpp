// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_source.hpp"

#include "openvino/core/except.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view STREAM_BAD_STATUS_MESSAGE = "The stream is in bad status";

/**
 * @return The size of the underlying buffer, from the beginning of the stream to its end.
 */
size_t get_stream_total_size(std::istream& stream) {
    OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

    const std::streampos backupCursor = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const std::streampos streamEnd = stream.tellg();
    stream.seekg(backupCursor, std::ios_base::beg);

    return streamEnd;
}

}  // namespace

namespace intel_npu {

BlobSource::BlobSource(std::istream& source, const ov::log::Level log_level = Logger::global().level())
    : BlobSource(source, log_level) {}

BlobSource::BlobSource(const ov::Tensor& source, const ov::log::Level log_level = Logger::global().level())
    : BlobSource(source, log_level) {}

BlobSource::BlobSource(const std::variant<std::reference_wrapper<std::istream>,
                                          std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>& source,
                       const ov::log::Level log_level)
    : m_source(source),
      m_logger(Logger("BlobSource", log_level)) {}

void BlobSource::copy_from_source(void* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
}

void* BlobSource::interpret_from_source(const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
}

ov::Tensor BlobSource::get_roi_tensor_from_source(const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
}

void BlobSource::move_cursor(const int64_t offset, const std::ios_base::seekdir reference) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);
        stream->get().seekg(std::streamoff(offset), reference);
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);
        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    OPENVINO_ASSERT(cursor <= m_size);

    switch (reference) {
    case std::ios::beg: {
        OPENVINO_ASSERT(offset >= 0);
        OPENVINO_ASSERT(offset <= m_size);
        cursor = offset;
        break;
    }
    case std::ios::cur: {
        if (offset >= 0) {
            OPENVINO_ASSERT(cursor <= cursor + offset);
            OPENVINO_ASSERT(cursor + offset <= m_size);
        } else {
            OPENVINO_ASSERT(cursor >= -offset);
        }

        cursor += offset;
        break;
    }
    case std::ios::end: {
        OPENVINO_ASSERT(offset <= 0);
        OPENVINO_ASSERT(m_size >= -offset);

        cursor = m_size + offset;
        break;
    }
    default: {
        OPENVINO_THROW("");
    }
    }
}

size_t BlobSource::get_cursor() const {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        return static_cast<size_t>(stream->get().tellg());
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    OPENVINO_ASSERT(cursor <= m_size);
    return cursor;
}

size_t BlobSource::get_total_size() const {
    return m_size;
}

size_t BlobSource::get_remaining_size() const {
    const size_t cursor = get_cursor();
    OPENVINO_ASSERT(cursor <= m_size);
    return m_size - cursor;
}

}  // namespace intel_npu
