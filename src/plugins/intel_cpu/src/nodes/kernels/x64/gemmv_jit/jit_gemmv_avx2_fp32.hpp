// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// Simple portable fallback (works on AVX2 and above). Not JIT.
class RefGemmvFp32 : public GemmvKernel {
public:
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "ref_fp32"; }
};

} // namespace ov::intel_cpu::x64::gemmv_jit
