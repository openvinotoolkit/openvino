// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "emitters/utils.hpp"
#include "libxsmm.h"

namespace ov::intel_cpu::tpp::utils {

// Note: The macro allows to automatically set appropriate environment variables for TPP/Libxsmm kernel compilation
// All TPP kernels must be compiled using this macro.
// * LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX enables more accurate exp approximation and exact division in TPP
// * LIBXSMM_GEMM_K_A_PF_DIST allows to tweak prefetching for GEMM kernels
#define COMPILE_TPP_KERNEL(...)                                          \
    [&]() {                                                              \
        setenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX", "1", 1); \
        setenv("LIBXSMM_GEMM_K_A_PF_DIST", "4", 1);                      \
        auto res = reinterpret_cast<const uintptr_t>(__VA_ARGS__);       \
        unsetenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX");       \
        unsetenv("LIBXSMM_GEMM_K_A_PF_DIST");                            \
        return res;                                                      \
    }()

inline libxsmm_datatype ov_to_xsmm_dtype(ov::element::Type_t element_type) {
    switch (element_type) {
    case ov::element::Type_t::f32:
        return LIBXSMM_DATATYPE_F32;
    case ov::element::Type_t::bf16:
        return LIBXSMM_DATATYPE_BF16;
    case ov::element::Type_t::f16:
        return LIBXSMM_DATATYPE_F16;
    case ov::element::Type_t::i8:
        return LIBXSMM_DATATYPE_I8;
    case ov::element::Type_t::u8:
        return LIBXSMM_DATATYPE_U8;
    default:
        OV_CPU_JIT_EMITTER_THROW("Attempt to convert unsupported ov data type:", element_type);
        return LIBXSMM_DATATYPE_IMPLICIT;
    }
}

}  // namespace ov::intel_cpu::tpp::utils
