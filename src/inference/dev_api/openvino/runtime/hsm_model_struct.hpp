// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

/** @brief Sends raw, unframed IR XML bytes directly to the destination sink without buffering. */
OPENVINO_RUNTIME_API void encode_model_struct(std::string_view xml, const SectionSink& sink);

/** @brief Produces XML on demand, when the writer finalizes the section. */
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
