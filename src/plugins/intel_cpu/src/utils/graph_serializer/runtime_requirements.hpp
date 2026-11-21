// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "config.h"
#include "openvino/core/model.hpp"

namespace ov::intel_cpu {

static constexpr uint64_t runtime_requirements_magic = 0x4F564350555F5252ULL;  // "OVCPU_RR" in ASCII
static constexpr uint32_t runtime_requirements_version = 1;
static constexpr uint64_t runtime_requirements_max_size = 4096;

/**
 * @brief Builds the runtime requirements descriptor of a compiled model.
 *
 * The descriptor lists only what the blob actually depends on, so that it can be reused across
 * OpenVINO versions and CPUs whenever possible:
 *   meta=2.0;ov=<producer version>;arch=<cpu architecture>;ops=<non-standard ops>;caps=<host capabilities>
 *
 * - ov   - producer OpenVINO version. The blob can be imported by the same or a newer runtime.
 * - arch - CPU architecture of the producer. Must match.
 * - ops  - internal (non-standard opset) operations serialized into the blob, grouped by opset:
 *          <opset>::<op>[@<schema>]+<op>...,<opset>::...  Every op must be known by the consumer with the same
 *          schema version (see internal_op_schema_versions() in runtime_requirements.cpp).
 * - caps - host capabilities the graph needs, each op requires only what it really uses. "ov." capabilities are
 *          checked with the OpenVINO CPU feature detection, "dnnl." capabilities with oneDNN. Examples: ov.avx2 for
 *          RMS, ov.amx_bf16 for LLMMLP and QKVProjection, dnnl.avx2 for compressed matmuls. A snippets Subgraph
 *          needs dnnl.avx2 on x86_64, dnnl.asimd on arm64 and ov.rvv on riscv64.
 *
 * @param model  Transformed model which is serialized into the blob.
 * @param config Configuration the model was transformed with.
 */
std::string build_runtime_requirements(const std::shared_ptr<const ov::Model>& model, const Config& config);

/**
 * @brief Checks the runtime requirements descriptor against the current runtime and host.
 * @return Empty string if compatible, otherwise the reason of incompatibility.
 */
std::string check_runtime_requirements(const std::string& requirements);

bool is_runtime_requirements_compatible(const std::string& requirements);

}  // namespace ov::intel_cpu
