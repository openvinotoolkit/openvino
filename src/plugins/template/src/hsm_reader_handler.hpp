// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Reader-side HSM logic for this plugin: HsmReaderHandler is the ISectionReaderHandler that claims every
// section of this blob (Core-owned and TEMPLATE-owned alike) during
// ov::runtime::hsm::Reader::read_sections() dispatch - decoding is entirely internal to it, nothing is
// exposed as a free function.

#pragma once

#include <map>
#include <optional>
#include <string>

#include "hsm_common.hpp"
#include "openvino/core/any.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::template_plugin {

class HsmReaderHandler final : public runtime::hsm::ISectionReaderHandler {
public:
    bool handle_section(const runtime::hsm::ManifestEntry& entry, const runtime::hsm::Section& section) override;

    void check_mandatory_sections() const;

    std::shared_ptr<ov::AlignedBuffer> source_blob;  // Set before read_sections() when importing ov::Tensor
    std::optional<uint64_t> model_id;
    std::optional<std::string> runtime_requirements;
    std::shared_ptr<ov::AlignedBuffer> model;
    std::shared_ptr<ov::AlignedBuffer> weights;
    std::optional<hsm::BlobInfo> blob_info;
    ov::AnyMap compiled_options;

private:
    using Decoder = void (*)(HsmReaderHandler&, const runtime::hsm::Section&);
    static const std::map<uint32_t, Decoder> s_decoders;
};

}  // namespace ov::template_plugin
