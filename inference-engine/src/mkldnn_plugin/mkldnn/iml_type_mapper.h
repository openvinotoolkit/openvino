// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace MKLDNNPlugin {

enum impl_desc_type {
    unknown = 0x00000000,
    undef,
    // Optimization approach
    simple = 1<<6,
    ref    = 1<<7,
    jit    = 1<<8,
    gemm   = 1<<9,
    // CPU version
    sse42  = 1<<10,
    avx    = 1<<11,
    avx2   = 1<<12,
    avx512 = 1<<13,
    blas   = 1<<14,
    any    = 1<<15,
    uni    = 1<<16,
    // Other specificator
    _1x1    = 1<<17,
    _dw     = 1<<18,
    // Other info
    reorder = 1<<19,
    // winograd
    winograd = 1<<20,
    // real types
    ref_any             = ref  | any,

    gemm_any            = gemm | any,
    gemm_blas           = gemm | blas,
    gemm_avx512         = gemm | avx512,
    gemm_avx2           = gemm | avx2,
    gemm_avx            = gemm | avx,
    gemm_sse42          = gemm | sse42,

    jit_avx512_winograd = jit  | avx512 | winograd,
    jit_avx512          = jit  | avx512,
    jit_avx2            = jit  | avx2,
    jit_avx             = jit  | avx,
    jit_sse42           = jit  | sse42,
    jit_uni             = jit  | uni,

    jit_avx512_1x1      = jit  | avx512 | _1x1,
    jit_avx2_1x1        = jit  | avx2   | _1x1,
    jit_avx_1x1         = jit  | avx    | _1x1,
    jit_sse42_1x1       = jit  | sse42  | _1x1,
    jit_uni_1x1         = jit  | uni    | _1x1,

    jit_avx512_dw       = jit  | avx512 | _dw,
    jit_avx2_dw         = jit  | avx2   | _dw,
    jit_avx_dw          = jit  | avx    | _dw,
    jit_sse42_dw        = jit  | sse42  | _dw,
    jit_uni_dw          = jit  | uni    | _dw,
};

impl_desc_type parse_impl_name(std::string impl_desc_name);

}  // namespace MKLDNNPlugin
