// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
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
    /// so the sections are mapped from it instead of copied.
    explicit HsmReaderHandler(std::shared_ptr<ov::AlignedBuffer> source_blob = {});

    bool handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) override;

    /// Throws if the blob lacks a mandatory section.
    void check_mandatory_sections() const;

    std::optional<std::string> runtime_requirements;
    std::optional<uint64_t> model_id;                 //!< See hsm::model_id;
    std::shared_ptr<ov::AlignedBuffer> model_struct;  //!< See hsm::model_struct.
    std::shared_ptr<ov::AlignedBuffer> weights;       //!< See hsm::weights; null if it is absent or empty.

private:
    using Decoder = void (*)(HsmReaderHandler&, const runtime::hsm::Section&);
    static const std::map<uint32_t, Decoder> s_decoders;

    std::shared_ptr<ov::AlignedBuffer> m_source_blob;
};

}  // namespace ov::intel_cpu
