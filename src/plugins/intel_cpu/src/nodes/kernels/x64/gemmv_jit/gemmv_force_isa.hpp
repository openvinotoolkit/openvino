// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::intel_cpu::x64::gemmv_jit {

// Canonical list of override targets exposed to tooling/benchmarks.
enum class gemmv_force_isa_t {
    auto_mode = 0,
    avx512_fp32,
    avx2_fp32,
    ref_fp32,
    vnni,
    amx_int8,
    amx_bf16,
};

// Returns the current global override (AUTO by default).
gemmv_force_isa_t get_gemmv_force_isa();

// Installs a new override. Intended for benchmarking/test harnesses only.
void set_gemmv_force_isa(gemmv_force_isa_t mode);

// Helpers for tooling.
const char* gemmv_force_isa_to_cstr(gemmv_force_isa_t mode);
gemmv_force_isa_t gemmv_force_isa_from_cstr(const char* value);

} // namespace ov::intel_cpu::x64::gemmv_jit
