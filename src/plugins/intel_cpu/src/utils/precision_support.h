// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

bool hasHardwareSupport(const ov::element::Type& precision);
ov::element::Type defaultFloatPrecision();

/**
 * @brief Whether fp8 (f8e4m3 / f8e5m2) weights may be kept compressed down to the
 *        oneDNN primitive and decompressed inside the kernel, for the given
 *        activation precision.
 *
 * Mirrors oneDNN's own ISA gate for the brgemm_matmul `is_bf16_fp8` / `is_f16_fp8`
 * configurations (`brgemm_matmul_utils.cpp`):
 *   - bf16: AVX512-AMX (Sapphire Rapids) / AMX-FP16 (Granite Rapids) / AVX10.2 (Nova
 *     Lake and later)
 *   - f16: AMX-FP16 / AVX10.2 only - Sapphire Rapids is excluded, it has no native
 *     f16 compute
 *   - anything else (f32, dynamic, ...): false - oneDNN has no such fp8
 *     configuration, so the fp8 constant must keep being folded. In particular
 *     `ov::element::dynamic` is not a wildcard here: it is the real value
 *     `Config::inferencePrecision` takes in ACCURACY execution mode (no forced
 *     bf16/f16 promotion), and in that mode fp8 weights must fold just like f32.
 *
 * No default argument on purpose: callers that only want to know "is fp8 weights
 * decompression available on this platform at all", without a specific activation
 * precision in hand yet (e.g. before widening the accepted activation set for
 * pattern matching), should pass `ov::element::bf16` explicitly - bf16 is supported
 * wherever any of the three platforms is present, so it is exactly that query.
 */
bool hasFp8WeightsDecompressionSupport(ov::element::Type activationPrecision);

}  // namespace ov::intel_cpu
