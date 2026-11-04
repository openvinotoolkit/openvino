// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Writer-side HSM logic for this plugin: HsmWriterHandler is the ISectionWriterHandler that contributes every
// section of the blob (Core-owned and TEMPLATE-owned alike) during ov::runtime::hsm::IWriter::add_sections().

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string_view>

#include "config.hpp"
#include "hsm_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/hsm_writer.hpp"

namespace ov::template_plugin {

class HsmWriterHandler final : public runtime::hsm::ISectionWriterHandler {
public:
    /// Keeps @p model and streams its serialized form into the blob at IWriter::finalize(), so the structure and the
    /// weights are never copied into temporary buffers.
    /// @p cache_options are the compile options that form the Core cache ID; they are persisted and hashed with the
    /// model.
    HsmWriterHandler(const std::shared_ptr<ov::Model>& model,
                     const Configuration& cfg,
                     const ov::AnyMap& cache_options,
                     std::string_view runtime_requirements);

    /// The handler must outlive IWriter::finalize(): inline sections are borrowed from its encoders.
    void handle_section(runtime::hsm::IWriter& writer) const override;

private:
    class BytesEncoder;

    struct Slot {
        Slot(const hsm::SectionId& id,
             const std::shared_ptr<BytesEncoder>& bytes,
             runtime::hsm::SectionAlignment align = {});
        Slot(const hsm::SectionId& id,
             runtime::hsm::SectionEncoderPtr encoder,
             std::optional<size_t> size,
             runtime::hsm::SectionAlignment align);

        bool add_to(runtime::hsm::IWriter& writer) const;

        hsm::SectionId m_id;
        runtime::hsm::SectionAlignment m_align;
        runtime::hsm::SectionEncoderPtr m_encoder;
        std::optional<size_t> m_size;  // unset: the size is known only after encoding
        ov::util::MemoryView m_view;   // set for the sections small enough to be borrowed as inline
    };

    std::map<uint32_t, Slot> m_encoders;
};

}  // namespace ov::template_plugin
