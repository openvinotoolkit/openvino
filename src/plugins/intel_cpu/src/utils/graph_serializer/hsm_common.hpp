// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Wire contract of the CPU HSM blob, shared by HsmReaderHandler and HsmWriterHandler. Keep it free of reader and
// writer code - the two handlers must stay independent. `hsm` is this plugin's own namespace; the runtime API is
// spelled runtime::hsm.

#pragma once

#include <cstdint>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::intel_cpu::hsm {

inline constexpr runtime::hsm::DeviceId device_id = runtime::hsm::cpu_device_id;

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

/// Core-owned: the string checked by is_runtime_requirements_compatible() on import.
// TODO CVS-196454: use the common Runtime Requirements codec instead of the plain string.
inline constexpr SectionId runtime_requirements{runtime::hsm::any_device_id, runtime::hsm::runtime_requirements_tag};
/// Layout 1: the whole legacy ModelSerializer stream (DataHeader, custom data, constants, XML), unchanged.
// TODO CVS-196450, CVS-196451: replace with the common Model Struct and Weights sections.
// TODO CVS-196456, CVS-196452: add the common Model Id and Compiled Options sections; CPU keeps its options in the
// model rt_info until then.
inline constexpr SectionId serialized_model{device_id,
                                            runtime::hsm::SectionTag::make_device_tag(0, /*is_inline=*/false)};

}  // namespace ov::intel_cpu::hsm
