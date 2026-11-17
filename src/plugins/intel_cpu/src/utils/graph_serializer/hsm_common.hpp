// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Wire contract of the CPU HSM blob, shared by HsmReaderHandler and HsmWriterHandler. Keep it free of reader and
// writer code - the two handlers must stay independent. `hsm` is this plugin's own namespace; the runtime API is
// spelled runtime::hsm.
//
// Layout of the container, in file order (the legacy stream was [DataHeader][Custom data][Blobs][IR] behind a
// requirements prefix; the manifest now replaces the DataHeader, and the custom data is no longer written):
//
//   [ HSM header           ]  32 bytes
//   [ runtime_requirements ]  common section, checked first so an incompatible blob is rejected before any payload
//   [ padding              ]
//   [ weights              ]  page-aligned offset and slot size; may be empty (no constants, or weightless cache)
//   [ model_struct         ]  page-aligned offset and slot size; IR XML, encrypted when cache encryption is on
//   [ manifest             ]  one entry per section, starts on a page boundary
//
// The XML refers to constants by offset from the start of the weights section.

#pragma once

#include <cstdint>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::intel_cpu::hsm {

inline constexpr runtime::hsm::DeviceId device_id = runtime::hsm::cpu_device_id;

struct SectionId {
    runtime::hsm::DeviceId device{};
    runtime::hsm::SectionTag tag{};
};

constexpr uint32_t section_key(runtime::hsm::DeviceId device, runtime::hsm::SectionTag tag) noexcept {
    return (static_cast<uint32_t>(device) << 24) | (static_cast<uint32_t>(tag.value[0]) << 16) |
           (static_cast<uint32_t>(tag.value[1]) << 8) | static_cast<uint32_t>(tag.value[2]);
}

constexpr uint32_t section_key(const SectionId& id) noexcept {
    return section_key(id.device, id.tag);
}

/// Core-owned: the string checked by is_runtime_requirements_compatible() on import.
// CVS-196454: use the common Runtime Requirements codec instead of the plain string.
inline constexpr SectionId runtime_requirements{runtime::hsm::any_device_id, runtime::hsm::runtime_requirements_tag};
/// Core-owned, inline.
// The id algorithm is not defined yet, the writer stores a placeholder and the reader does not use it.
inline constexpr SectionId model_id{runtime::hsm::any_device_id, runtime::hsm::model_id_tag};
/// The compiled model's IR XML (encrypted when cache encryption is on), without constants.
// CVS-196450: replace with the common Model Struct section once its payload is defined.
// CVS-196452: add the common Compiled Options section; CPU keeps its options in the model rt_info until then.
inline constexpr SectionId model_struct{device_id, runtime::hsm::model_tag};
// CVS-196451: replace with the common Weights section once its payload is defined.
inline constexpr SectionId weights{device_id, runtime::hsm::weights_tag};

}  // namespace ov::intel_cpu::hsm
