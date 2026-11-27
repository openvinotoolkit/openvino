// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Wire contract of the HETERO HSM blob, shared by HsmReaderHandler and HsmWriterHandler. Keep it free of reader and
// writer code - the two handlers must stay independent. `hsm` is this plugin's own namespace; the runtime API is
// spelled runtime::hsm.
//
// The container itself (header, manifest, section tags, device ids, compatibility rules) is defined by
// src/inference/dev_api/openvino/runtime/hsm_format.hpp; the legacy HETERO layout it replaces is described in
// blob_serialization.hpp. This file only defines what HETERO stores in the sections.
//
// Layout of the container, in file order:
//
//   [ HSM header ]
//   [ runtime_requirements ]  common section, checked first so an incompatible blob is rejected before any payload
//   [ metadata ]              hetero XML: name, I/O mapping, hetero config, one <compiled_submodel> per submodel
//   per submodel i, in order, either
//     [ submodel_blob ]       page-aligned; the blob exported by the sub-device, opaque to HETERO
//   or
//     [ submodel_ir_weights ] page-aligned; IR fallback for devices without export/import, may be empty
//     [ submodel_ir_xml ]
//   [ manifest ]
//
// Submodel sections carry the submodel index in ManifestEntry::tag_reserved (see decode_section_index()). Every
// submodel index of the metadata must have exactly one of the two variants above.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::hetero::hsm {

inline constexpr runtime::hsm::DeviceId device_id = runtime::hsm::hetero_device_id;

struct SectionId {
    runtime::hsm::DeviceId device{};
    runtime::hsm::SectionTag tag{};
};

constexpr uint32_t section_key(runtime::hsm::DeviceId device, uint32_t tag_id) noexcept {
    return (static_cast<uint32_t>(device) << 24) | tag_id;
}

constexpr uint32_t section_key(const SectionId& id) noexcept {
    return section_key(id.device, id.tag.id());
}

/// Core-owned: the string checked by check_hsm_runtime_requirements() on import.
// TODO CVS-196454: use the common Runtime Requirements codec instead of the plain string.
inline constexpr SectionId runtime_requirements{runtime::hsm::any_device_id, runtime::hsm::runtime_requirements_tag};
/// The hetero metadata XML.
inline constexpr SectionId metadata{device_id, runtime::hsm::model_tag};

/// Device-local ids of the HETERO sections - wire values: never change or reuse one once shipped.
/// They are turned into section tags by runtime::hsm::SectionTag::make_device_tag(), see hsm_format.hpp.
enum class Tags : uint32_t {
    submodel_blob = 0,        //!< See #submodel_blob.
    submodel_ir_xml = 1,      //!< See #submodel_ir_xml.
    submodel_ir_weights = 2,  //!< See #submodel_ir_weights.
};

constexpr SectionId make_section_id(Tags tag) noexcept {
    return {device_id, runtime::hsm::SectionTag::make_device_tag(static_cast<uint32_t>(tag), false)};
}

/// The blob exported by the sub-device of the submodel given by the section index.
inline constexpr SectionId submodel_blob = make_section_id(Tags::submodel_blob);
/// IR xml of a submodel whose device cannot export a blob.
inline constexpr SectionId submodel_ir_xml = make_section_id(Tags::submodel_ir_xml);
/// IR weights of the same submodel.
inline constexpr SectionId submodel_ir_weights = make_section_id(Tags::submodel_ir_weights);

inline std::array<uint8_t, 4> encode_section_index(uint32_t index) noexcept {
    std::array<uint8_t, 4> bytes{};
    std::memcpy(bytes.data(), &index, sizeof(index));
    return bytes;
}

inline uint32_t decode_section_index(const std::array<uint8_t, 4>& bytes) noexcept {
    uint32_t index = 0;
    std::memcpy(&index, bytes.data(), sizeof(index));
    return index;
}

inline constexpr uint32_t requirements_format_version = 1;

}  // namespace ov::hetero::hsm
