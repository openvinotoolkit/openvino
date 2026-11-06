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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

/** @brief Sends exactly @p xml to the destination sink without buffering or adding framing. */
OPENVINO_RUNTIME_API void encode_model_struct(std::string_view xml, const SectionSink& sink);

/** @brief Produces the same raw XML bytes on demand, possibly in multiple sink calls. */
class IModelStructSource {
public:
    virtual ~IModelStructSource() = default;
    virtual void write_structure(const SectionSink& sink) = 0;
};

/** @brief Produces XML directly into the section sink at IWriter::finalize() time. */
class ModelStructEncoder final : public ISectionEncoder {
public:
    explicit ModelStructEncoder(std::shared_ptr<IModelStructSource> source) : m_source{std::move(source)} {}

    void encode(const SectionSink& sink) const override {
        m_source->write_structure(sink);
    }

private:
    std::shared_ptr<IModelStructSource> m_source;
};

/** @brief Copies raw XML bytes into a string; returns nullopt for an invalid nonempty view. */
OPENVINO_RUNTIME_API std::optional<std::string> decode_model_struct(ov::util::MemoryView payload);

/** @brief Decodes a buffer- or stream-backed section; returns nullopt if it cannot be read. */
OPENVINO_RUNTIME_API std::optional<std::string> decode_model_struct(const Section& section) noexcept;

}  // namespace v1
}  // namespace ov::runtime::hsm
