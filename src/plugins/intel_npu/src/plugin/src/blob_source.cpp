// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_source.hpp"

#include "openvino/core/except.hpp"

namespace {

constexpr std::string_view STREAM_BAD_STATUS_MESSAGE = "The stream is in bad status";
constexpr std::string_view STREAM_READ_WITHOUT_COPY_MESSAGE =
    "The underlying data type of the blob source is a stream. Cannot read from a stream without copying the data.";
constexpr std::string_view INVALID_CURSOR_MESSAGE =
    "The cursor of the blob source points outside its designated buffer";
constexpr std::string_view INVALID_MOVE_MESSAGE =
    "Attempted to move the blob source cursor outside its designated buffer";
constexpr std::string_view LOGGER_NAME = "BlobSource";
constexpr size_t STARTING_CURSOR_POSITION = 0;

/**
 * @return The size of the underlying buffer, from the beginning of the stream to its end.
 */
size_t get_stream_total_size(std::istream& stream) {
    OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

    const std::streampos backupCursor = stream.tellg();
    OPENVINO_ASSERT(backupCursor != std::streampos(-1), STREAM_BAD_STATUS_MESSAGE);

    stream.seekg(0, std::ios_base::end);
    OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

    const std::streampos streamEnd = stream.tellg();
    OPENVINO_ASSERT(streamEnd != std::streampos(-1), STREAM_BAD_STATUS_MESSAGE);
    OPENVINO_ASSERT(streamEnd >= backupCursor, "Invalid stream size");

    stream.seekg(backupCursor, std::ios_base::beg);
    OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

    return static_cast<size_t>(streamEnd);
}

}  // namespace

namespace intel_npu {

BlobSource::BlobSource(std::istream& source, const ov::log::Level log_level)
    : m_source(source),
      m_size(get_stream_total_size(source)),
      m_logger(Logger(LOGGER_NAME.data(), log_level)) {
    m_logger.debug("Initialized a BlobSource using a stream object");
}

BlobSource::BlobSource(const ov::Tensor& source, const ov::log::Level log_level)
    : m_source(std::make_pair<>(std::cref(source), STARTING_CURSOR_POSITION)),
      m_size(source.get_byte_size()),
      m_logger(Logger(LOGGER_NAME.data(), log_level)) {
    OPENVINO_ASSERT(source.get_shape().size() == 1 && source.get_shape().at(0) == source.get_byte_size(),
                    "Only streams and unidimensional tensors are supported as blob sources");
    m_logger.debug("Initialized a BlobSource using a tensor object");
}

void BlobSource::read_into_buffer(void* destination, const size_t size) {
    m_logger.trace("Copying %zu bytes", size);

    const size_t remaining = get_remaining_size();
    OPENVINO_ASSERT(size <= remaining,
                    "Attempted to read ",
                    size,
                    " bytes from the blob but only ",
                    remaining,
                    " bytes remain");

    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream->get(), STREAM_BAD_STATUS_MESSAGE);
        stream->get().read(reinterpret_cast<char*>(destination), size);
        OPENVINO_ASSERT(stream->get(), STREAM_BAD_STATUS_MESSAGE);
        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    std::memcpy(destination, tensor.get().data<const char>() + cursor, size);
    cursor += size;
}

const void* BlobSource::read_view(const size_t size) {
    m_logger.trace("Reading %zu bytes without copying", size);
    OPENVINO_ASSERT(!std::get_if<std::reference_wrapper<std::istream>>(&m_source), STREAM_READ_WITHOUT_COPY_MESSAGE);

    const size_t remaining = get_remaining_size();
    OPENVINO_ASSERT(size <= remaining,
                    "Attempted to read ",
                    size,
                    " bytes from the blob but only ",
                    remaining,
                    " bytes remain");

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    cursor += size;
    return tensor.get().data<const char>() + cursor - size;
}

ov::Tensor BlobSource::create_roi_tensor(const size_t size) {
    m_logger.trace("Creating an roi tensor of %zu bytes without copying", size);
    OPENVINO_ASSERT(!std::get_if<std::reference_wrapper<std::istream>>(&m_source), STREAM_READ_WITHOUT_COPY_MESSAGE);

    const size_t remaining = get_remaining_size();
    OPENVINO_ASSERT(size <= remaining,
                    "Attempted to read ",
                    size,
                    " bytes from the blob but only ",
                    remaining,
                    " bytes remain");

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    cursor += size;
    return ov::Tensor(tensor, ov::Coordinate{cursor - size}, ov::Coordinate{cursor});
}

void BlobSource::seekg(const int64_t offset, const std::ios_base::seekdir reference) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream->get(), STREAM_BAD_STATUS_MESSAGE);
        stream->get().seekg(std::streamoff(offset), reference);
        OPENVINO_ASSERT(stream->get(), STREAM_BAD_STATUS_MESSAGE);
        return;
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    OPENVINO_ASSERT(cursor <= m_size, INVALID_CURSOR_MESSAGE);

    switch (reference) {
    case std::ios::beg: {
        OPENVINO_ASSERT(offset >= 0 && static_cast<size_t>(offset) <= m_size, INVALID_MOVE_MESSAGE);
        cursor = static_cast<size_t>(offset);
        break;
    }
    case std::ios::cur: {
        OPENVINO_ASSERT(
            offset > 0 ? static_cast<size_t>(offset) <= m_size - cursor : static_cast<size_t>(-offset) <= cursor,
            INVALID_MOVE_MESSAGE);
        cursor += offset;
        break;
    }
    case std::ios::end: {
        OPENVINO_ASSERT(offset <= 0 && static_cast<size_t>(-offset) <= m_size, INVALID_MOVE_MESSAGE);
        cursor = m_size + offset;
        break;
    }
    default: {
        OPENVINO_THROW("A request to move the blob source cursor was made, but the given reference is invalid");
    }
    }
}

size_t BlobSource::tellg() const {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        OPENVINO_ASSERT(stream->get(), STREAM_BAD_STATUS_MESSAGE);

        const std::streampos cursor = stream->get().tellg();
        OPENVINO_ASSERT(cursor != std::streampos(-1), STREAM_BAD_STATUS_MESSAGE);
        return static_cast<size_t>(cursor);
    }

    auto& [tensor, cursor] = std::get<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(m_source);
    OPENVINO_ASSERT(cursor <= m_size, INVALID_CURSOR_MESSAGE);
    return cursor;
}

size_t BlobSource::get_total_size() const {
    return m_size;
}

size_t BlobSource::get_remaining_size() const {
    return m_size - tellg();
}

bool BlobSource::is_contiguous() const {
    // The buffer behind a "stream" object is not guaranteed to be contiguous
    return !std::get_if<std::reference_wrapper<std::istream>>(&m_source);
}

}  // namespace intel_npu
