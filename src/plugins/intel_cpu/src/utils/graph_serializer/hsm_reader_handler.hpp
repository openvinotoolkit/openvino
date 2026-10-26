// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace ov::intel_cpu {

class HsmReaderHandler final : public runtime::hsm::ISectionReaderHandler {
public:
    /// @p source_blob keeps the memory behind the sections alive. Pass it when the Reader was opened over a buffer,
    /// so the model is mapped from it instead of copied.
    explicit HsmReaderHandler(std::shared_ptr<ov::AlignedBuffer> source_blob = {});

    /// Unrecognized sections are skipped. Throws if a recognized section is duplicated or cannot be read.
    bool handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) override;

    /// Throws if the blob lacks a mandatory section or was written with a newer, unsupported layout.
    void check_mandatory_sections() const;

    std::optional<std::string> runtime_requirements;
    std::shared_ptr<ov::AlignedBuffer> serialized_model;  //!< The legacy stream, see hsm::serialized_model.

private:
    using Decoder = void (*)(HsmReaderHandler&, const runtime::hsm::Section&);
    static const std::map<uint32_t, Decoder> s_decoders;

    std::shared_ptr<ov::AlignedBuffer> m_source_blob;
};

}  // namespace ov::intel_cpu
