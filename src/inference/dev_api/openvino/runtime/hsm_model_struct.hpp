// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Codec for the compiled model structure stored in a pointer-mode #model_tag section.
 * @file openvino/runtime/hsm_model_struct.hpp
 *
 * @par Section
 * - Identified by `(device, #model_tag)`, where `device` is the device that produced the structure.
 * - #ManifestEntry::tag_reserved and #ManifestEntry::pointer_reserved are zero.
 * - #ManifestEntry::offset is chosen by the writer from the alignment passed to IWriter::add_section(); the
 *   codec requires none. Padding after the payload is not part of it.
 *
 * @par Payload
 * @verbatim
   ManifestEntry::offset                                           offset + size
   |<---------------------- ManifestEntry::size ---------------------->|
   | structure bytes in the device's own format, exactly as produced   |
   @endverbatim
 * - Core defines only the framing: the payload is the producer's bytes, unchanged, with no magic, version,
 *   length prefix, checksum or terminator added; its length is #ManifestEntry::size.
 * - The format of the bytes is owned by the producing device and may be text or binary, e.g. IR XML for
 *   Template and CPU, a device-specific binary for GPU or NPU. The device is also responsible for versioning
 *   that format and for how it references weights stored in other sections or files.
 * - A zero-size payload is valid and decodes to an empty buffer; whether it is acceptable is up to the device.
 *
 * @par Encoding and decoding
 * - The encoder writes the producer's buffer unchanged, at IWriter::finalize() time.
 * - The decoder returns the payload bytes unchanged without interpreting them; they are untrusted input and
 *   must be validated by the device that parses them.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/** @brief Generates the structure with the producer at IWriter::finalize() time and writes it to the section.
 * @note The whole structure is held in memory while it is written. For a device that streams a large structure
 * today (e.g. GPU with weights in the program) this duplicates it in host memory; such a device needs a stream
 * writer variant that forwards to the SectionSink directly (TODO). */
class ModelStructEncoder final : public ISectionEncoder {
public:
    /** @brief Returns the whole structure; a retried finalize() calls it again, so it must return the same bytes.
     * Return AlignedBuffer() for an empty structure - AlignedBuffer(0) holds one byte. */
    using Producer = std::function<std::shared_ptr<ov::AlignedBuffer>()>;

    explicit ModelStructEncoder(Producer producer) : m_producer{std::move(producer)} {
        OPENVINO_ASSERT(m_producer, "HSM model struct encoder requires a producer");
    }

    void encode(const SectionSink& sink) const override {
        const auto structure = m_producer();
        OPENVINO_ASSERT(structure, "HSM model struct producer returned no buffer");
        sink({structure->get_ptr<std::byte>(), structure->size()});
    }

private:
    Producer m_producer;
};

/** @brief Returns the section as a buffer: a zero-copy view when it is addressable, valid as long as the
 * container passed to Reader::open() is alive, otherwise an owned copy; `std::nullopt` if it cannot be read. */
OPENVINO_RUNTIME_API std::optional<std::shared_ptr<ov::AlignedBuffer>> decode_model_struct(const Section& section);

}  // namespace v1
}  // namespace ov::runtime::hsm
