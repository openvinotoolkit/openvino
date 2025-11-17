// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gemmv_ukernel.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

// Micro-kernel wrapper around AMX INT8 GEMMV implementation.
class JitGemmvAmxInt8Kernel final : public GemmvKernel {
public:
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "amx_int8"; }
};

// Micro-kernel wrapper around AMX BF16 GEMMV implementation.
class JitGemmvAmxBf16Kernel final : public GemmvKernel {
public:
    void operator()(const gemmv_ukr_params_t* p) const override;
    const char* name() const override { return "amx_bf16"; }
};

} // namespace ov::intel_cpu::x64::gemmv_jit
