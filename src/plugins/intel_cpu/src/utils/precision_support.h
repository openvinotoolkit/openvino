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
 *        activation precision. Mirrors oneDNN's own ISA gate for the brgemm_matmul
 *        `is_bf16_fp8` / `is_f16_fp8` configurations. `ov::element::dynamic` is a
 *        real, valid input here (ACCURACY execution mode), not a wildcard - it
 *        correctly resolves to false, same as f32.
 */
bool hasFp8WeightsDecompressionSupport(ov::element::Type activationPrecision);

/** FP16 compressed-weight dispatch validated on the AVX512-FP16 path without AMX-FP16. */
bool hasFp16WeightsDecompressionSupport();

}  // namespace ov::intel_cpu
