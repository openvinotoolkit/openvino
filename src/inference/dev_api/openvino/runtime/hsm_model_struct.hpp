// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Codec for serialized IR topology stored under the pointer-mode #model_tag.
 * @file openvino/runtime/hsm_model_struct.hpp
 *
 * The payload is exactly the IR XML bytes, in their original order. Its length is the section size in the
 * manifest; there is no length prefix, trailing NUL, or other framing. Weights belong in a separate section.
 * This payload contract applies to plugins that store IR XML under #model_tag (such as Template); other
 * devices may use the same tag for their own serialized model format. The decoder returns the XML bytes
 * as a string without parsing or validating them.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/** @brief Generates the structure with the producer at IWriter::finalize() time and writes it to the section. */
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

/** @brief Copies raw XML bytes into a string; returns nullopt for an invalid nonempty view. */
OPENVINO_RUNTIME_API std::optional<std::string> decode_model_struct(ov::util::MemoryView payload);

/** @brief Decodes a buffer- or stream-backed section; returns nullopt if it cannot be read. */
OPENVINO_RUNTIME_API std::optional<std::string> decode_model_struct(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
