// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Wire contract of the TEMPLATE HSM blob (device id, section ids, BlobInfo), shared by HsmReaderHandler and
// HsmWriterHandler. Keep it free of reader and writer code - the two handlers must stay independent.
// `hsm` is this plugin's own namespace; the runtime API is spelled runtime::hsm.

#pragma once

#include <cstdint>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::template_plugin {
namespace hsm {

// Placeholder until the HSM device catalog gives TEMPLATE its own id (private range starts at 0x80).
inline constexpr runtime::hsm::DeviceId device_id = runtime::hsm::private_device_id_range_start;
struct SectionId {
    runtime::hsm::DeviceId device;
    runtime::hsm::SectionTag tag;
};

constexpr uint32_t section_key(runtime::hsm::DeviceId device, runtime::hsm::SectionTag tag) noexcept {
    return (static_cast<uint32_t>(device) << 24) | (static_cast<uint32_t>(tag.value[0]) << 16) |
           (static_cast<uint32_t>(tag.value[1]) << 8) | static_cast<uint32_t>(tag.value[2]);
}

constexpr uint32_t section_key(const SectionId& id) noexcept {
    return section_key(id.device, id.tag);
}

inline constexpr SectionId model_id{runtime::hsm::any_device_id, runtime::hsm::model_id_tag};
inline constexpr SectionId runtime_requirements{runtime::hsm::any_device_id, runtime::hsm::runtime_requirements_tag};
inline constexpr SectionId model_struct{device_id, runtime::hsm::model_tag};
inline constexpr SectionId weights{device_id, runtime::hsm::weights_tag};
inline constexpr SectionId compiled_options{device_id, runtime::hsm::compiled_options_tag};
// TEMPLATE-specific: layout version and feature flags, no common equivalent.
inline constexpr SectionId blob_info{device_id, runtime::hsm::SectionTag::make_device_tag(3, /*is_inline=*/true)};

/// Inline section: layout version of this plugin's own sections plus feature flags.
struct BlobInfo {
    static constexpr uint32_t current_layout_version = 1;
    static constexpr uint32_t weights_embedded = 1u << 0;
    static constexpr uint32_t weightless = 1u << 1;  //!< Model needs no weights at all - not just "stored externally".

    uint32_t layout_version = current_layout_version;
    uint32_t flags = 0;
};
static_assert(sizeof(BlobInfo) == 8, "BlobInfo must fit the inline slot without padding");

}  // namespace hsm
}  // namespace ov::template_plugin
