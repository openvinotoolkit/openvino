// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/rv64/cpu_isa_traits.hpp>

namespace ov::intel_cpu::riscv64::brgemm_utils {

inline bool is_fp32_supported() {
    return dnnl::impl::cpu::rv64::mayiuse(dnnl::impl::cpu::rv64::v);
}

inline bool is_bf16_supported() {
    return dnnl::impl::cpu::rv64::mayiuse(dnnl::impl::cpu::rv64::zvfbfwma);
}

inline bool is_fp16_supported() {
    return dnnl::impl::cpu::rv64::mayiuse(dnnl::impl::cpu::rv64::zvfh);
}

}  // namespace ov::intel_cpu::riscv64::brgemm_utils
