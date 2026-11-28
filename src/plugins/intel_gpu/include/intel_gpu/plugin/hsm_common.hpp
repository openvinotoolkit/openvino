// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Wire contract of the GPU HSM blob, shared by the reader and writer handlers.
//
// Layout of the container, in file order:
//
//   [ HSM header           ]  32 bytes
//   [ runtime_requirements ]  common tag, checked before the model payload is read
//   [ compile_options      ]  common codec; stores the GPU cache mode
//   [ padding              ]
//   [ model                ]  page-aligned GPU section: the legacy compiled-model payload (inputs, outputs, graph,
//                             kernels, constants) without its cache-mode and requirements prefix; ciphertext
//                             only when cache encryption is on (length from the manifest)
//   [ manifest             ]  one entry per section, including inline GPU payload_info
//
// Constants inside the model section are page-aligned relative to the section start, so a page-aligned blob in
// memory can be used for zero-copy weights.

#pragma once

#include <cstdint>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::intel_gpu::hsm {

struct SectionId {
    runtime::hsm::DeviceId device{};
    runtime::hsm::SectionTag tag{};
};

/// Inline GPU section, PayloadInfo.
inline constexpr SectionId payload_info{runtime::hsm::gpu_device_id, runtime::hsm::SectionTag::make_device_tag(0, true)};
/// The string checked by CompiledModel::is_runtime_requirements_compatible() on import.
// CVS-196454: use the common Runtime Requirements codec instead of the plain string.
inline constexpr SectionId runtime_requirements{runtime::hsm::any_device_id, runtime::hsm::runtime_requirements_tag};
/// Persisted GPU compile options needed to interpret the model section.
inline constexpr SectionId compile_options{runtime::hsm::gpu_device_id, runtime::hsm::compile_options_tag};
/// Opaque compiled graph, including embedded constants when not using external weights.
// CVS-196450: replace with the common Model Struct section once its payload is defined.
inline constexpr SectionId model{runtime::hsm::gpu_device_id, runtime::hsm::model_tag};

inline constexpr uint32_t encrypted_flag = 1;

/// Describes how the model section is stored.
struct PayloadInfo {
    static constexpr uint32_t current_version = 2;
    uint32_t version = current_version;
    uint32_t flags = 0;       // encrypted_flag
    uint32_t requirements_version = 0;
    uint32_t reserved32 = 0;
    uint64_t reserved = 0;
};
static_assert(sizeof(PayloadInfo) == 24);

}  // namespace ov::intel_gpu::hsm
